from typing import Callable
from collections import OrderedDict
import numpy as np

import torch
from torch import autograd
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

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


class OPOLO(TorchBaseAlgorithm):
    """
    Class that implements OPOLO training.
    https://arxiv.org/abs/2102.13185
    """

    def __init__(
        self,
        mode: str,  # airl, gail, or fairl
        nu_net: nn.Module,
        discriminator: nn.Module,
        expert_replay_buffer,
        policy_lr: float = 1e-3,
        disc_lr: float = 1e-3,
        nu_lr: float = 1e-3,
        discount: float = 0.99,
        use_grad_pen: bool = True,
        grad_pen_weight: float = 10.0,
        num_update_loops_per_train_call: int = 1,
        num_disc_updates_per_loop_iter: int = 1,
        num_policy_updates_per_loop_iter: int = 100,
        policy_optim_batch_size: int = 1024,
        disc_optim_batch_size: int = 1024,
        policy_mean_reg_weight: float = 1e-3,
        policy_std_reg_weight: float = 1e-3,
        optimizer_class: Callable[..., optim.Optimizer] = optim.Adam,
        beta_1: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mode = mode
        self.discriminator = discriminator
        self.nu_net = nu_net
        self.expert_replay_buffer = expert_replay_buffer

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.discount = discount
        self.policy_optim_batch_size = policy_optim_batch_size
        self.disc_optim_batch_size = disc_optim_batch_size

        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(ptu.device)

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.policy_optimizer = optimizer_class(
            self.exploration_policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.disc_optimizer = optimizer_class(
            self.discriminator.parameters(), lr=disc_lr, betas=(beta_1, 0.999)
        )
        self.nu_optimizer = optimizer_class(
            self.nu_net.parameters(), lr=nu_lr, betas=(beta_1, 0.999)
        )

        self.disc_eval_statistics = None

    def get_batch(self, batch_size, from_expert, keys=None):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _end_epoch(self):
        self.disc_eval_statistics = None
        super()._end_epoch()

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.disc_eval_statistics)
        super().evaluate(epoch)

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_disc_updates_per_loop_iter):
                self._do_reward_training(epoch)
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)

    def _do_reward_training(self, epoch):
        """
        Train the discriminator
        """
        self.disc_optimizer.zero_grad()

        keys = ["observations", "next_observations"]
        if self.wrap_absorbing:
            keys.append("absorbing")

        expert_batch = self.get_batch(self.disc_optim_batch_size, True, keys)
        policy_batch = self.get_batch(self.disc_optim_batch_size, False, keys)

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        if self.wrap_absorbing:
            # pass
            expert_obs = torch.cat(
                [expert_obs, expert_batch["absorbing"][:, 0:1]], dim=-1
            )
            policy_obs = torch.cat(
                [policy_obs, policy_batch["absorbing"][:, 0:1]], dim=-1
            )

        expert_next_obs = expert_batch["next_observations"]
        policy_next_obs = policy_batch["next_observations"]
        if self.wrap_absorbing:
            # pass
            expert_next_obs = torch.cat(
                [expert_next_obs, expert_batch["absorbing"][:, 1:]], dim=-1
            )
            policy_next_obs = torch.cat(
                [policy_next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
            )
        expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
        policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)

        disc_logits = self.discriminator(disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            self.disc_eval_statistics["Disc Acc"] = np.mean(ptu.get_numpy(accuracy))
            if self.use_grad_pen:
                self.disc_eval_statistics["Disc Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Disc Grad Pen W"] = np.mean(
                    self.grad_pen_weight
                )

    def _do_policy_training(self, epoch):
        """
        Update policy and nu_net with dice objective
        """
        self.policy_optimizer.zero_grad()
        self.nu_optimizer.zero_grad()

        keys = ["observations", "next_observations", "actions"]
        batch = self.get_batch(self.policy_optim_batch_size, False, keys)

        obs = initial_obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # Get discriminator rewards
        self.discriminator.eval()
        disc_input = torch.cat([obs, next_obs], dim=1)
        disc_logits = self.discriminator(disc_input).detach()
        self.discriminator.train()

        # compute the reward using the algorithm
        if self.mode == "airl":
            # If you compute log(D) - log(1-D) then you just get the logits
            disc_rewards = disc_logits
        elif self.mode == "gail":  # -log (1-D) > 0
            disc_rewards = F.softplus(
                disc_logits, beta=1
            )  # F.softplus(disc_logits, beta=-1)
        elif self.mode == "gail2":  # log D < 0
            disc_rewards = F.softplus(
                disc_logits, beta=-1
            )  # F.softplus(disc_logits, beta=-1)
        elif self.mode == "fairl":  # fairl
            disc_rewards = torch.exp(disc_logits) * (-1.0 * disc_logits)
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        initial_actions = self.exploration_policy(initial_obs)[0]
        (
            next_actions,
            next_action_mean,
            next_action_log_std,
        ) = self.exploration_policy(next_obs)[:3]

        init_inputs = torch.cat([initial_obs, initial_actions], dim=1)
        inputs = torch.cat([obs, actions], dim=1)
        next_inputs = torch.cat([next_obs, next_actions], dim=1)

        nu_0 = self.nu_net(init_inputs)
        nu = self.nu_net(inputs)
        nu_next = self.nu_net(next_inputs)

        linear_loss = nu_0.mean()

        bellman_diff = nu - (self.discount * nu_next + disc_rewards)
        non_linear_loss = (
            weighted_softmax(bellman_diff, ptu.ones_like(bellman_diff), dim=0).detach()
            * bellman_diff
        ).sum()

        loss = non_linear_loss - linear_loss

        mean_reg_loss = self.policy_mean_reg_weight * (next_action_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (next_action_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss

        nu_loss = loss
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
                    ptu.get_numpy(next_action_mean),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(next_action_log_std),
                )
            )

    @property
    def networks(self):
        return [self.exploration_policy, self.nu_net, self.discriminator]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(nu_net=self.nu_net)
        return snapshot

    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)
