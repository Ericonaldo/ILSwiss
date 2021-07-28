import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm


class BC(TorchBaseAlgorithm):
    def __init__(
        self,
        mode,  # 'MLE' or 'MSE'
        expert_replay_buffer,
        num_updates_per_train_call=1,
        batch_size=1024,
        lr=1e-3,
        momentum=0.0,
        optimizer_class=optim.Adam,
        **kwargs
    ):
        assert mode in ["MLE", "MSE"], "Invalid mode!"
        if kwargs["wrap_absorbing"]:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.expert_replay_buffer = expert_replay_buffer

        self.batch_size = batch_size

        self.optimizer = optimizer_class(
            self.exploration_policy.parameters(), lr=lr, betas=(momentum, 0.999)
        )

        self.num_updates_per_train_call = num_updates_per_train_call

    def get_batch(self, batch_size, keys=None, use_expert_buffer=True):
        if use_expert_buffer:
            rb = self.expert_replay_buffer
        else:
            rb = self.replay_buffer
        batch = rb.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _do_training(self, epoch):
        for t in range(self.num_updates_per_train_call):
            self._do_update_step(epoch, use_expert_buffer=True)

    def _do_update_step(self, epoch, use_expert_buffer=True):
        batch = self.get_batch(
            self.batch_size,
            keys=["observations", "actions"],
            use_expert_buffer=use_expert_buffer,
        )

        obs = batch["observations"]
        acts = batch["actions"]

        self.optimizer.zero_grad()
        if self.mode == "MLE":
            log_prob = self.exploration_policy.get_log_prob(obs, acts)
            loss = -1.0 * log_prob.mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["Log-Likelihood"] = ptu.get_numpy(-1.0 * loss)
        else:
            pred_acts = self.exploration_policy(obs)[0]
            squared_diff = (pred_acts - acts) ** 2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["MSE"] = ptu.get_numpy(loss)
        loss.backward()
        self.optimizer.step()

    @property
    def networks(self):
        return [self.exploration_policy]
