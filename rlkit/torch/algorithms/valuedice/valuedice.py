from typing import Callable
from collections import OrderedDict
import numpy as np

import torch
from torch import autograd
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm


def weighted_softmax(x, weights, dim=0):
    x = x - torch.max(x, dim=dim)[0]
    return (
        weights
        * torch.exp(x)
        / torch.sum(weights * torch.exp(x), dim=dim, keepdims=True)
    )


class ValueDice(TorchBaseAlgorithm):
    """Class that implements DualDICE training."""

    def __init__(
        self,
        nu_net: nn.Module,
        expert_replay_buffer,
        policy_lr: str = 1e-3,
        nu_lr: str = 1e-3,
        discount: float = 0.99,
        replay_reg: float = 0.05,
        use_grad_pen: bool = True,
        grad_pen_weight: float = 10.0,
        num_update_loops_per_train_call: int = 1,
        expert_batch_size: int = 1024,
        replay_batch_size: int = 1024,
        policy_mean_reg_weight: float = 1e-3,
        policy_std_reg_weight: float = 1e-3,
        optimizer_class: Callable[..., optim.Optimizer] = optim.Adam,
        beta_1: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nu_net = nu_net
        self.expert_replay_buffer = expert_replay_buffer

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.discount = discount
        self.replay_reg = replay_reg
        self.expert_batch_size = expert_batch_size
        self.replay_batch_size = replay_batch_size
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call

        self.policy_optimizer = optimizer_class(
            self.exploration_policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.nu_optimizer = optimizer_class(
            self.nu_net.parameters(), lr=nu_lr, betas=(beta_1, 0.999)
        )

    def get_batch(self, batch_size, from_expert, keys=None):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            self._do_dice_training(epoch)

    def _do_dice_training(self, epoch):
        """
        Update policy and nu_net with dice objective
        """
        self.policy_optimizer.zero_grad()
        self.nu_optimizer.zero_grad()

        keys = ["observations", "next_observations", "actions"]
        expert_batch = self.get_batch(self.expert_batch_size, True, keys)
        replay_batch = self.get_batch(self.replay_batch_size, False, keys)

        expert_obs = expert_initial_obs = expert_batch["observations"]
        expert_actions = expert_batch["actions"]
        expert_next_obs = expert_batch["next_observations"]
        replay_obs = replay_batch["observations"]
        replay_actions = replay_batch["actions"]
        replay_next_obs = replay_batch["next_observations"]

        policy_initial_actions = self.exploration_policy(expert_initial_obs)[0]
        policy_next_actions = self.exploration_policy(expert_next_obs)[0]
        (
            replay_next_actions,
            replay_next_action_mean,
            replay_next_action_log_std,
            replay_log_pi,
        ) = self.exploration_policy(replay_next_obs, return_log_prob=True)[:4]

        expert_init_inputs = torch.cat(
            [expert_initial_obs, policy_initial_actions], dim=1
        )
        expert_inputs = torch.cat([expert_obs, expert_actions], dim=1)
        expert_next_inputs = torch.cat([expert_next_obs, policy_next_actions], dim=1)

        replay_inputs = torch.cat([replay_obs, replay_actions], dim=1)
        replay_next_inputs = torch.cat([replay_next_obs, replay_next_actions], dim=1)

        expert_nu_0 = self.nu_net(expert_init_inputs)
        expert_nu = self.nu_net(expert_inputs)
        expert_nu_next = self.nu_net(expert_next_inputs)

        replay_nu = self.nu_net(replay_inputs)
        replay_nu_next = self.nu_net(replay_next_inputs)

        expert_diff = expert_nu - self.discount * expert_nu_next
        replay_diff = replay_nu - self.discount * replay_nu_next

        linear_loss_expert = (expert_nu_0 * (1 - self.discount)).mean()
        linear_loss_replay = replay_diff.mean()

        linear_loss = (
            linear_loss_expert * (1 - self.replay_reg)
            + linear_loss_replay * self.replay_reg
        )

        replay_expert_diff = torch.cat([expert_diff, replay_diff], dim=0)
        replay_expert_weights = torch.cat(
            [
                ptu.ones_like(expert_diff) * (1 - self.replay_reg),
                ptu.ones_like(replay_diff) * self.replay_reg,
            ],
            dim=0,
        )
        replay_expert_weights = replay_expert_weights / replay_expert_weights.sum()

        non_linear_loss = (
            weighted_softmax(replay_expert_diff, replay_expert_weights, dim=0).detach()
            * replay_expert_diff
        ).sum()

        loss = non_linear_loss - linear_loss

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            nu_interp = eps * expert_inputs + (1 - eps) * replay_inputs
            nu_next_interp = eps * expert_next_inputs + (1 - eps) * replay_next_inputs
            nu_interp = torch.cat([nu_interp, nu_next_interp], dim=0)
            nu_interp = nu_interp.detach()
            nu_interp.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.nu_net(nu_interp).sum(),
                inputs=[nu_interp],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            nu_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight

        mean_reg_loss = (
            self.policy_mean_reg_weight * (replay_next_action_mean**2).mean()
        )
        std_reg_loss = (
            self.policy_std_reg_weight * (replay_next_action_log_std**2).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss

        nu_loss = loss + nu_grad_pen_loss
        policy_loss = -loss + policy_reg_loss

        nu_loss.backward(retain_graph=True, inputs=list(self.nu_net.parameters()))
        policy_loss.backward(inputs=list(self.exploration_policy.parameters()))

        self.policy_optimizer.step()
        self.nu_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Nu Loss"] = np.mean(ptu.get_numpy(nu_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics["Linear Loss"] = np.mean(ptu.get_numpy(linear_loss))
            self.eval_statistics["Non-Linear Loss"] = np.mean(
                ptu.get_numpy(non_linear_loss)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy mu",
                    ptu.get_numpy(replay_next_action_mean),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(replay_next_action_log_std),
                )
            )
            if self.use_grad_pen:
                self.eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

    @property
    def networks(self):
        return [self.exploration_policy, self.nu_net]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(nu_net=self.nu_net)
        return snapshot
