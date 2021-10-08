from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm


class TD3(Trainer):
    """
    Twin Delayed Deep Deterministic policy gradients

    https://arxiv.org/abs/1802.09477
    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        reward_scale=1.0,
        discount=0.99,
        target_policy_noise=0.2,
        target_policy_noise_clip=0.5,
        policy_lr=1e-3,
        qf_lr=1e-3,
        policy_and_target_update_period=2,
        soft_target_tau=0.005,
        qf_criterion=None,
        optimizer_class=optim.Adam,
        **kwargs
    ):
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy

        self.reward_scale = reward_scale
        self.discount = discount

        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.policy_and_target_update_period = policy_and_target_update_period
        self.soft_target_tau = soft_target_tau
        self.qf_criterion = qf_criterion

        self.target_policy = self.policy.copy()
        self.target_qf1 = self.qf1.copy()
        self.target_qf2 = self.qf2.copy()
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.eval_statistics = None

        self._n_train_steps_total = 0

    def train_step(self, batch):

        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Critic operations.
        """
        policy_outputs = self.target_policy(next_obs)
        noisy_next_actions = policy_outputs[0]

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = rewards + (1.0 - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        policy_actions = policy_loss = None

        if self._n_train_steps_total % self.policy_and_target_update_period == 0:

            policy_outputs = self.policy(obs, deterministic=True)
            policy_actions = policy_outputs[0]
            q_output = self.qf1(obs, policy_actions)
            policy_loss = -q_output.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self._update_target_network()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            if policy_loss is None:
                # policy_loss = 0
                policy_outputs = self.policy(obs, deterministic=True)
                policy_actions = policy_outputs[0]
                q_output = self.qf1(obs, policy_actions)
                policy_loss = -q_output.mean()

            self.eval_statistics = OrderedDict()
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Targets",
                    ptu.get_numpy(q_target),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Bellman Errors 1",
                    ptu.get_numpy(bellman_errors_1),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Bellman Errors 2",
                    ptu.get_numpy(bellman_errors_2),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy Action",
                    ptu.get_numpy(policy_actions),
                )
            )
        self._n_train_steps_total += 1

    def _update_target_network(self):
        ptu.soft_update_from_to(self.policy, self.target_policy, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_policy=self.target_policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
        return snapshot

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
