from collections import OrderedDict
from itertools import count
import time
from typing import Dict, List, Tuple, Union, Type

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core import logger
from rlkit.torch.common.networks import BNN


class BNNTrainer(Trainer):
    def __init__(
        self,
        bnn: BNN,
        lr: float = 1e-3,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        fc_weight_decays: Union[float, List] = None,
        num_elites: int = None,
        reward_scale: float = 1.0,
        batch_size: int = 32,
        max_epochs: int = None,  # if not setting max_epochs, training will stop following the early stopping scheme
        max_epochs_since_update: int = 5,  # for early stopping
        max_grad_steps: int = None,  # max gradient steps
        holdout_ratio: float = 0.0,
        max_holdout: int = 5000,
        log_freq: int = 100,  # log frequency, -1 for silence
        timer=None,
        max_t: float = None,  # time limit of training
        **kwargs,
    ):
        self.bnn = bnn
        self.fc_weight_decays = fc_weight_decays
        num_hidden = self.bnn.num_layers - 1
        if self.fc_weight_decays is None:
            self.fc_weight_decays = (
                [2.5e-5, 5e-5] + [7.5e-5] * (num_hidden - 2) + [1e-4]
                if num_hidden > 2
                else [2.5e-5, 1e-4]
            )
        elif isinstance(self.fc_weight_decays, float):
            self.fc_weight_decays = [self.fc_weight_decays] * (num_hidden + 1)
        elif isinstance(self.fc_weight_decays, list):
            assert len(self.fc_weight_decays) == num_hidden + 1
        self.num_elites = self.bnn.num_nets if num_elites is None else num_elites
        self.num_elites = min(self.num_elites, self.bnn.num_nets)

        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_epochs_since_update = max_epochs_since_update
        self.max_grad_steps = max_grad_steps
        self.holdout_ratio = holdout_ratio
        self.max_holdout = max_holdout
        self.log_freq = log_freq
        self.timer = timer
        self.max_t = max_t
        self.eval_statistics = None

        opt_param_groups = [
            {"params": fc.parameters(), "weight_decay": wd}
            for fc, wd in zip(self.bnn.layers, self.fc_weight_decays)
        ]
        opt_param_groups += [{"params": [self.bnn.min_log_var, self.bnn.max_log_var]}]
        self.optimizer = optimizer_class(opt_param_groups, lr=lr)

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, add_var_loss: bool = True
    ) -> torch.Tensor:
        mean, log_var = self.bnn(inputs, ret_log_var=True)
        inv_var = torch.exp(-log_var)
        if add_var_loss:
            loss = torch.mean((mean - targets) ** 2 * inv_var, dim=[-2, -1])
            var_loss = torch.mean(log_var, dim=[-2, -1])
            loss = loss + var_loss
        else:
            loss = torch.mean((mean - targets) ** 2, dim=[-2, -1])
        return loss

    def train_step(self, batch: torch.Tensor):
        self._epochs_since_update = 0
        # net_id: [state_dict_i for i in layer_id]
        self._state: Dict[int, List[OrderedDict[str : torch.Tensor]]] = {}
        # net_id: (best_epoch, best_loss)
        self._snapshots = {i: (None, 1e10) for i in range(self.bnn.num_nets)}

        obs = batch["observations"]
        actions = batch["actions"]
        rewards = self.reward_scale * batch["rewards"]
        # terminals = batch["terminals"]
        next_obs = batch["next_observations"]
        inputs = torch.cat((obs, actions), dim=-1)
        targets = torch.cat((rewards, next_obs - obs), dim=-1)

        num_holdout: int = min(
            int(inputs.shape[0] * self.holdout_ratio), self.max_holdout
        )
        perm: np.ndarray = np.random.permutation(inputs.shape[0])

        ho_inputs = inputs[perm[:num_holdout]]
        ho_inputs: torch.Tensor = torch.tile(
            ho_inputs[None], dims=[self.bnn.num_nets, 1, 1]
        )
        inputs = inputs[perm[num_holdout:]]
        ho_targets = targets[perm[:num_holdout]]
        ho_targets: torch.Tensor = torch.tile(
            ho_targets[None], dims=[self.bnn.num_nets, 1, 1]
        )
        targets = targets[perm[num_holdout:]]

        logger.log(f"BNN | Training {inputs.shape} | Holdout: {ho_inputs.shape}")
        inp_mean = torch.mean(inputs, dim=0, keepdim=True)
        inp_std = torch.std(inputs, dim=0, keepdim=True)
        inp_std[inp_std < 1e-12] = 1.0
        self.normalizer.set_mean(ptu.get_numpy(inp_mean))
        self.normalizer.set_std(ptu.get_numpy(inp_std))

        idxs: np.ndarray = np.random.randint(
            inputs.shape[0], size=[self.bnn.num_nets, inputs.shape[0]]
        )
        epoch_iter = range(self.max_epochs) if self.max_epochs else count()
        t_start = time.time()
        grad_updates = 0
        break_train = False
        for epoch in epoch_iter:
            num_batches = int(np.ceil(idxs.shape[-1] / self.batch_size))
            for batch_num in range(num_batches):
                batch_idx: np.ndarray = idxs[
                    :,
                    batch_num
                    * self.batch_size : min(
                        (batch_num + 1) * self.batch_size, idxs.shape[-1]
                    ),
                ]
                batch_idx = ptu.from_numpy(batch_idx, requires_grad=False).to(int)
                batch_inp = ptu.tf_like_gather(inputs, batch_idx)
                batch_tar = ptu.tf_like_gather(targets, batch_idx)

                mse_loss = torch.mean(
                    self.compute_loss(batch_inp, batch_tar, add_var_loss=True)
                )
                loss = (
                    mse_loss
                    + 0.01 * torch.mean(self.bnn.max_log_var)
                    - 0.01 * torch.mean(self.bnn.min_log_var)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                grad_updates += 1

            idxs = self._shuffle_rows(idxs)

            with torch.no_grad():
                eval_inp = ptu.tf_like_gather(inputs, idxs[:, : self.max_holdout])
                eval_tar = ptu.tf_like_gather(targets, idxs[:, : self.max_holdout])
                train_mse = self.compute_loss(eval_inp, eval_tar, add_var_loss=False)
            if epoch % self.log_freq == 0:
                logger.log(
                    f"BNN | epoch {epoch} | BNN Train MSE: {np.mean(ptu.get_numpy(train_mse))}"
                )
            if self.holdout_ratio > 1e-8:  # holdout
                with torch.no_grad():
                    holdout_mse = ptu.get_numpy(
                        self.compute_loss(ho_inputs, ho_targets, add_var_loss=False)
                    )
                if epoch % self.log_freq == 0:
                    logger.log(
                        f"BNN | epoch {epoch} | BNN Holdout MSE: {np.mean(holdout_mse)}"
                    )
                break_train = self._save_best(epoch, holdout_mse)

            t = time.time() - t_start
            if break_train or (
                self.max_grad_steps and grad_updates > self.max_grad_steps
            ):
                break
            if self.max_t and t > self.max_t:
                logger.log(
                    f"BNN | epoch {epoch} | Breaking because of timeout: {t} (max {self.max_t})"
                )
                break
        self._stamp("bnn_train")

        self._set_state()
        self._stamp("bnn_set_state")

        with torch.no_grad():
            final_holdout_mse = ptu.get_numpy(
                self.compute_loss(ho_inputs, ho_targets, add_var_loss=False)
            )
        self._stamp("bnn_holdout")

        ho_idx: np.array = np.argsort(final_holdout_mse)
        self._model_idx = ho_idx[: self.num_elites].tolist()
        logger.log(
            f"BNN | Using {self.num_elites}/{self.bnn.num_nets} models: {self._model_idx}"
        )
        self._stamp("bnn_end")

        final_holdout_mse.sort()
        val_loss = np.mean(final_holdout_mse[: self.num_elites])
        logger.log(
            f"BNN | Holdout loss {final_holdout_mse} | Validation loss: {val_loss}"
        )
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics["BNN Loss"] = val_loss

    @property
    def normalizer(self):
        return self.bnn.normalizer

    @property
    def networks(self):
        return self.bnn.layers

    def get_snapshot(self):
        return dict(bnn=self.bnn)

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None

    def predict(
        self, input: torch.Tensor, factored: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bnn.predict(input, factored)

    def _stamp(self, s: str) -> None:
        if self.timer:
            self.timer.stamp(s)

    def _shuffle_rows(self, a: np.ndarray) -> np.ndarray:
        new_idx = np.argsort(np.random.uniform(size=a.shape), axis=-1)
        return a[np.arange(a.shape[0])[:, None], new_idx]

    def _save_state(self, net_id: int) -> None:
        self._state[net_id] = [
            OrderedDict({"weight": fc.weight.data, "bias": fc.bias.data})
            for fc in self.bnn.layers
        ]

    def _set_state(self):
        for layer_id in range(self.bnn.num_layers):
            for net_id in range(self.bnn.num_nets):
                layer = self.bnn.layers[layer_id]
                layer.load_state_dict(self._state[net_id][layer_id])

    def _save_best(self, epoch: int, holdout_mse: np.ndarray) -> bool:
        # update and early stopping
        upd = False
        for net_id in range(len(holdout_mse)):
            cur = holdout_mse[net_id]
            _, best = self._snapshots[net_id]
            imp = (best - cur) / best
            if imp > 0.01:
                self._snapshots[net_id] = (epoch, cur)
                self._save_state(net_id)
                upd = True
        if upd:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        return self._epochs_since_update > self.max_epochs_since_update

    def get_random_model_index(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self._model_idx, size=batch_size)

    def to(self, device):
        self.bnn.to(device)
