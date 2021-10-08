import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm


def calculate_log_ab(log_pis, log_trs, dim_c, device=ptu.device):  # forward-backward
    log_alpha = [
        torch.empty(dim_c, dtype=torch.float32, device=device).fill_(-math.log(dim_c))
    ]
    for log_tr, log_pi in zip(log_trs, log_pis):
        log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) + log_tr).logsumexp(
            dim=0
        ) + log_pi
        log_alpha.append(log_alpha_t)
    log_alpha = log_alpha[1:]

    log_beta = [torch.zeros(dim_c, dtype=torch.float32, device=device)]
    for log_tr, log_pi in zip(reversed(log_trs), reversed(log_pis)):
        log_beta_t = ((log_beta[-1] + log_pi).unsqueeze(dim=0) + log_tr).logsumexp(
            dim=-1
        )
        log_beta.append(log_beta_t)
    log_beta.reverse()
    log_beta = log_beta[1:]

    log_alpha = torch.stack(log_alpha)
    log_beta = torch.stack(log_beta)
    return log_alpha, log_beta


class OptionBC(TorchBaseAlgorithm):
    def __init__(
        self,
        mode,  # 'MLE' or 'MSE'
        expert_replay_buffer,
        num_updates_per_train_call=1,
        batch_size=1024,
        lr=1e-3,
        momentum=0.9,
        optimizer_class=optim.Adam,
        factor_ent=1.0,
        n_part=5,
        **kwargs
    ):
        assert mode in ["MSE", "MLE", "MAP", "MAP_5"], "Invalid mode!"
        if kwargs["wrap_absorbing"]:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.expert_replay_buffer = expert_replay_buffer
        self.batch_size = batch_size
        self.factor_ent = factor_ent
        self.n_part = n_part

        self.optimizer = optimizer_class(
            self.exploration_policy.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
            weight_decay=1.0e-3,
        )

        self.num_updates_per_train_call = num_updates_per_train_call

    def get_batch(self, batch_size, keys=None, use_expert_buffer=True):
        if use_expert_buffer:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def get_batch_trajs(self, traj_num=1, keys=None, use_expert_buffer=True):
        if use_expert_buffer:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.sample_trajs(traj_num, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _do_training(self, epoch):
        for t in range(self.num_updates_per_train_call):
            self._do_update_step(epoch, use_expert_buffer=True)

    def _do_update_step(self, epoch, use_expert_buffer=True):
        traj_batch = self.get_batch_trajs(
            self.batch_size,
            keys=["observations", "actions"],
            use_expert_buffer=use_expert_buffer,
        )

        obs_trajs = traj_batch["observations"]
        acts_trajs = traj_batch["actions"]
        demo_len = obs_trajs.shape[1]

        self.optimizer.zero_grad()
        if self.mode == "MLE":
            loss = 0
            for i in range(self.batch_size):
                obs_traj, acts_traj = obs_trajs[i], acts_trajs[i]
                log_pis = self.exploration_policy.log_prob_action(
                    obs_traj, None, acts_traj
                ).view(
                    -1, self.exploration_policy.dim_c
                )  # demo_len x ct
                log_trs = self.exploration_policy.log_trans(
                    obs_traj, None
                )  # demo_len x (ct_1 + 1) x ct
                # The last option ct_1 is for option # in the initial state
                log_tr0 = log_trs[0, -1]  # (ct,)
                log_trs = log_trs[1:, :-1]  # (demo_len-1) x ct_1 x ct

                log_alpha = [log_tr0 + log_pis[0]]  # forward message, 1 x ct
                for log_tr, log_pi in zip(log_trs, log_pis[1:]):  # for all demos
                    log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) + log_tr).logsumexp(
                        dim=0
                    ) + log_pi  # logsumexp arrpox max
                    log_alpha.append(log_alpha_t)

                log_alpha = torch.stack(log_alpha)  # demo_len x ct
                entropy = (
                    -(log_trs * log_trs.exp()).sum(dim=-1).mean()
                )  # first sum to dim ct then compute mean
                log_p = (
                    log_alpha.softmax(dim=-1).detach() * log_alpha
                ).sum()  # weighted sums # EM
                loss += -log_p - self.factor_ent * entropy

            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["Log-Likelihood"] = ptu.get_numpy(-1.0 * loss)
        elif self.mode == "MSE":
            batch = self.get_batch(
                self.batch_size,
                keys=["observations", "actions"],
                use_expert_buffer=use_expert_buffer,
            )
            obs = batch["observations"]
            acts = batch["actions"]
            pred_acts = self.exploration_policy(obs)[0]
            squared_diff = (pred_acts - acts) ** 2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["MSE"] = ptu.get_numpy(loss)
        elif self.mode == "EM":
            loss = 0
            option_traj = (
                torch.arange(
                    self.exploration_policy.dim_c, dtype=torch.long, device=ptu.device
                )
                .view(1, self.exploration_policy.dim_c)
                .repeat(demo_len + 1, 1)
            )  # (demo_len+1) x option_dim
            option_traj[0] = -1
            for i in range(self.batch_size):
                obs_traj, acts_traj = obs_trajs[i], acts_trajs[i]

                log_pis = self.exploration_policy.log_prob_action(
                    obs_traj, option_traj, acts_traj
                ).view(
                    -1, self.exploration_policy.dim_c
                )  # demo_len x ct
                log_trs = self.exploration_policy.log_trans(
                    obs_traj, option_traj
                )  # demo_len x (ct_1 + 1) x ct
                log_alpha, log_beta = calculate_log_ab(
                    log_pis, log_trs, self.exploration_policy.dim_c, ptu.device
                )
                pi = (log_alpha + log_beta).softmax(dim=-1).detach()
                marg_a = torch.cat(
                    (torch.zeros_like(log_alpha[0:1]), log_alpha[:-1]), dim=0
                )
                tr = (
                    (
                        marg_a.unsqueeze(dim=-1)
                        + (log_beta + log_pis).unsqueeze(dim=-2)
                        + log_trs
                    )
                    .softmax(dim=-1)
                    .detach()
                )

                loss_pi = -(pi * log_pis).sum()
                loss_tr = -(tr * log_trs).sum()
                loss += loss_pi + loss_tr

        loss.backward()
        self.optimizer.step()

    @property
    def networks(self):
        return [self.exploration_policy]
