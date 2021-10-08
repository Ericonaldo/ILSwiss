from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer

"""
Contains Option PPO
"""


class OptionPPO(Trainer):
    def __init__(
        self,
        policy,
        vf,
        clip_eps=0.2,
        reward_scale=1.0,
        discount=0.99,
        policy_lr=1e-3,
        option_lr=1e-3,
        vf_lr=1e-3,
        gae_tau=0.9,
        value_l2_reg=1e-3,
        use_value_clip=False,
        beta_1=0.9,
        update_epoch=10,
        mini_batch_size=128,
        optimizer_class=optim.Adam,
        lambda_entropy_policy=0.0,
        lambda_entropy_option=1e-2,
        **kwargs,
    ):
        self.policy = policy
        self.vf = vf
        self.mini_batch_size = mini_batch_size
        self.reward_scale = reward_scale
        self.discount = discount
        self.clip_eps = clip_eps
        self.update_epoch = update_epoch
        self.use_value_clip = use_value_clip
        self.value_l2_reg = value_l2_reg
        self.lambda_entropy_policy = lambda_entropy_policy
        self.lambda_entropy_option = lambda_entropy_option

        self.gae_tau = gae_tau

        self.eval_statistics = None

        self.option_policy_optimizer = optimizer_class(
            self.policy.option_policy.parameters(), lr=option_lr, betas=(beta_1, 0.999)
        )
        if isinstance(self.policy.action_log_std, nn.ParameterList):  # if the action_log_std is not shared and is a ParameterList
            self.policy_optimizer = optimizer_class(
                list(self.policy.policy.parameters()) + list(self.policy.action_log_std.parameters()),
                lr=policy_lr,
                betas=(beta_1, 0.999),
            )
        else:
            self.policy_optimizer = optimizer_class(
                list(self.policy.policy.parameters()) + [self.policy.action_log_std],
                lr=policy_lr,
                betas=(beta_1, 0.999),
            )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(), lr=vf_lr, betas=(beta_1, 0.999)
        )

    def calc_adv(self, trajs, train_policy=True, train_option=True):
        obs_array = []
        option_array = []
        prev_option_array = []
        action_array = []
        return_array = []
        option_adv_array = []
        option_vel_array = []
        adv_array = []
        vel_array = []

        for traj in trajs:
            observations = traj["observations"]
            actions = traj["actions"]
            rewards = self.reward_scale * traj["rewards"]
            prev_options = traj["prev_options"]
            options = traj["options"]

            vc = self.vf(observations)  # N x dim_c
            if train_option:
                pc = self.policy.log_trans(
                    observations, prev_options
                ).exp()  # N x dim_c
                option_values = (vc * pc).sum(dim=-1, keepdim=True).detach()
            else:
                option_values = torch.zeros_like(rewards)

            values = (
                vc.gather(dim=-1, index=options.type(torch.long)).detach()
                if train_policy
                else torch.zeros_like(rewards)
            )

            option_deltas = torch.zeros_like(rewards)
            deltas = torch.zeros_like(rewards)
            option_advantages = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)

            option_prev_value = 0.0
            prev_value = 0.0
            option_prev_advantage = 0.0
            prev_advantage = 0.0

            for i in reversed(range(rewards.size(0))):
                option_deltas[i] = (
                    rewards[i] + self.discount * option_prev_value - option_values[i]
                )
                deltas[i] = rewards[i] + self.discount * prev_value - values[i]
                option_advantages[i] = (
                    option_deltas[i]
                    + self.discount * self.gae_tau * option_prev_advantage
                )
                advantages[i] = (
                    deltas[i] + self.discount * self.gae_tau * prev_advantage
                )

                option_prev_value = option_values[i, 0]
                option_prev_advantage = option_advantages[i, 0]
                prev_value = values[i, 0]
                prev_advantage = advantages[i, 0]

            returns = values + advantages
            advantages = (advantages - advantages.mean()) / advantages.std()
            option_advantages = (
                option_advantages - option_advantages.mean()
            ) / option_advantages.std()

            obs_array.append(observations)
            option_array.append(options)
            prev_option_array.append(prev_options)
            action_array.append(actions)
            return_array.append(returns)
            option_adv_array.append(option_advantages)
            adv_array.append(advantages)
            option_vel_array.append(option_values)
            vel_array.append(values)

        obs_array = torch.cat(obs_array, dim=0)
        option_array = torch.cat(option_array, dim=0)
        prev_option_array = torch.cat(prev_option_array, dim=0)
        action_array = torch.cat(action_array, dim=0)
        return_array = torch.cat(return_array, dim=0)
        option_adv_array = torch.cat(option_adv_array, dim=0)
        adv_array = torch.cat(adv_array, dim=0)
        option_vel_array = torch.cat(option_vel_array, dim=0)
        vel_array = torch.cat(vel_array, dim=0)

        return (
            obs_array,
            option_array,
            prev_option_array,
            action_array,
            return_array,
            option_adv_array,
            adv_array,
            option_vel_array,
            vel_array,
        )

    def train_step(self, trajs, train_option=True, train_policy=True):

        with torch.no_grad():
            (
                observations,
                options,
                prev_options,
                actions,
                returns,
                option_advantages,
                advantages,
                option_vel_array,
                vel_array,
            ) = self.calc_adv(
                trajs, train_policy=train_policy, train_option=train_option
            )
            fixed_option_log_p = (
                self.policy.get_log_prob_option(
                    observations, prev_options, options
                ).detach()
                if train_option
                else torch.zeros_like(option_advantages)
            )
            fixed_log_p = (
                self.policy.get_log_prob_action(observations, actions, options).detach()
                if train_policy
                else torch.zeros_like(advantages)
            )
            fixed_pc = (
                self.policy.log_trans(observations, prev_options).exp().detach()
                if train_option
                else torch.zeros_like(advantages)
            )

        for _ in range(self.update_epoch):

            inds = torch.randperm(observations.shape[0])

            for ind_b in inds.split(self.mini_batch_size):
                (
                    observation_b,
                    option_b,
                    prev_option_b,
                    action_b,
                    return_b,
                    option_advantage_b,
                    advantage_b,
                    fixed_option_log_b,
                    fixed_log_b,
                    fixed_pc_b,
                    fixed_option_v_b,
                    fixed_v_b,
                ) = (
                    observations[ind_b],
                    options[ind_b],
                    prev_options[ind_b],
                    actions[ind_b],
                    returns[ind_b],
                    option_advantages[ind_b],
                    advantages[ind_b],
                    fixed_option_log_p[ind_b],
                    fixed_log_p[ind_b],
                    fixed_pc[ind_b],
                    option_vel_array[ind_b],
                    vel_array[ind_b],
                )

                if train_option:
                    option_advantage_b = (
                        (option_advantage_b - option_advantage_b.mean())
                        / (option_advantage_b.std() + 1e-8)
                        if ind_b.size(0) > 1
                        else 0.0
                    )
                    (
                        option_log_prob,
                        option_entropy,
                    ) = self.policy.option_log_prob_entropy(
                        observation_b, prev_option_b, option_b
                    )

                    option_v_pred = (self.vf(observation_b) * fixed_pc_b).sum(
                        dim=-1, keepdim=True
                    )
                    if self.use_value_clip:
                        option_v_pred_clip = fixed_option_v_b + (
                            option_v_pred - fixed_option_v_b
                        ).clamp(-self.clip_eps, self.clip_eps)
                        option_vf_loss = torch.max(
                            (option_v_pred - return_b).pow(2),
                            (option_v_pred_clip - return_b).pow(2),
                        ).mean()
                    else:
                        option_vf_loss = (option_v_pred - return_b).pow(2).mean()

                    for param in self.vf.parameters():
                        option_vf_loss += param.pow(2).sum() * self.value_l2_reg

                    ratio = (option_log_prob - fixed_option_log_b).exp()
                    option_pg_loss = -torch.min(
                        option_advantage_b * ratio,
                        option_advantage_b
                        * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps),
                    ).mean()
                    loss = (
                        option_pg_loss
                        + option_vf_loss * 0.5
                        - self.lambda_entropy_option * option_entropy.mean()
                    )

                    self.option_policy_optimizer.zero_grad()
                    self.vf_optimizer.zero_grad()
                    loss.backward()
                    # after many experiments i find that do not clamp performs the best
                    # torch.nn.utils.clip_grad_norm_(self.policy.get_param(low_policy=not is_option), 0.5)
                    self.option_policy_optimizer.step()
                    self.vf_optimizer.step()

                if train_policy:
                    advantage_b = (
                        (advantage_b - advantage_b.mean()) / (advantage_b.std() + 1e-8)
                        if ind_b.size(0) > 1
                        else 0.0
                    )
                    log_prob, entropy = self.policy.policy_log_prob_entropy(
                        observation_b, option_b, action_b
                    )

                    v_pred = self.vf(observation_b).gather(
                        dim=-1, index=option_b.type(torch.long)
                    )
                    if self.use_value_clip:
                        v_pred_clip = fixed_v_b + (v_pred - fixed_v_b).clamp(
                            -self.clip_eps, self.clip_eps
                        )
                        vf_loss = torch.max(
                            (v_pred - return_b).pow(2), (v_pred_clip - return_b).pow(2)
                        ).mean()
                    else:
                        vf_loss = (v_pred - return_b).pow(2).mean()

                    for param in self.vf.parameters():
                        vf_loss += param.pow(2).sum() * self.value_l2_reg

                    ratio = (log_prob - fixed_log_b).exp()
                    pg_loss = -torch.min(
                        advantage_b * ratio,
                        advantage_b
                        * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps),
                    ).mean()
                    loss = (
                        pg_loss
                        + vf_loss * 0.5
                        - self.lambda_entropy_policy * entropy.mean()
                    )
                    self.policy_optimizer.zero_grad()
                    self.vf_optimizer.zero_grad()
                    loss.backward()
                    # after many experiments i find that do not clamp performs the best
                    # torch.nn.utils.clip_grad_norm_(self.policy.get_param(low_policy=not is_option), 0.5)
                    self.policy_optimizer.step()
                    self.vf_optimizer.step()

            """
            Save some statistics for eval
            """
            if self.eval_statistics is None:
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                self.eval_statistics = OrderedDict()
                self.eval_statistics["Reward Scale"] = self.reward_scale
                self.eval_statistics["High VF Loss"] = np.mean(
                    ptu.get_numpy(option_vf_loss)
                )
                self.eval_statistics["High PG Loss"] = np.mean(
                    ptu.get_numpy(option_pg_loss)
                )
                self.eval_statistics["Low VF Loss"] = np.mean(ptu.get_numpy(vf_loss))
                self.eval_statistics["Low PG Loss"] = np.mean(ptu.get_numpy(pg_loss))

    @property
    def networks(self):
        return [
            self.policy,
            self.vf,
        ]

    def get_snapshot(self):
        return dict(
            vf=self.vf,
            policy=self.policy,
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None

    def to(self, device):
        super.to(device)
