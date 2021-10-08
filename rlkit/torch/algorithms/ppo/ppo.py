from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer


class PPO(Trainer):
    def __init__(
        self,
        policy,
        vf,
        mini_batch_size=64,
        clip_eps=0.2,
        reward_scale=1.0,
        discount=0.99,
        policy_lr=3e-4,
        value_lr=3e-4,
        gae_tau=0.9,
        value_l2_reg=1e-3,
        use_value_clip=False,
        optimizer_class=optim.Adam,
        update_epoch=10,
        lambda_entropy_policy=0.0,
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
        self.lambda_entropy = lambda_entropy_policy

        self.gae_tau = gae_tau

        self.eval_statistics = None

        self.policy_optim = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.value_optim = optimizer_class(
            self.vf.parameters(),
            lr=value_lr,
        )

    def calc_adv(self, trajs):
        obs_array = []
        action_array = []
        return_array = []
        adv_array = []
        vel_array = []

        for traj in trajs:
            observations = traj["observations"]
            rewards = self.reward_scale * traj["rewards"]
            actions = traj["actions"]

            values = self.vf(observations).detach()
            deltas = torch.zeros_like(values)
            advantages = torch.zeros_like(values)

            prev_value = 0.0
            prev_advantage = 0.0

            for i in reversed(range(rewards.size(0))):
                deltas[i] = rewards[i] + self.discount * prev_value - values[i]
                advantages[i] = (
                    deltas[i] + self.discount * self.gae_tau * prev_advantage
                )

                prev_value = values[i, 0]
                prev_advantage = advantages[i, 0]

            returns = values + advantages
            advantages = (advantages - advantages.mean()) / advantages.std()

            obs_array.append(observations)
            action_array.append(actions)
            return_array.append(returns)
            adv_array.append(advantages)
            vel_array.append(values)

        obs_array = torch.cat(obs_array, dim=0)
        action_array = torch.cat(action_array, dim=0)
        return_array = torch.cat(return_array, dim=0)
        adv_array = torch.cat(adv_array, dim=0)
        vel_array = torch.cat(vel_array, dim=0)

        return obs_array, action_array, return_array, adv_array, vel_array

    def train_step(self, trajs):

        with torch.no_grad():
            (
                observations,
                actions,
                returns,
                advantages,
                vel_array,
            ) = self.calc_adv(trajs)
            fixed_log_probs = self.policy.get_log_prob(observations, actions).detach()

        for _ in range(self.update_epoch):

            inds = torch.randperm(observations.shape[0])

            for ind_b in inds.split(self.mini_batch_size):

                (
                    observation_b,
                    action_b,
                    return_b,
                    advantages_b,
                    fixed_log_b,
                    fixed_v_b,
                ) = (
                    observations[ind_b],
                    actions[ind_b],
                    returns[ind_b],
                    advantages[ind_b],
                    fixed_log_probs[ind_b],
                    vel_array[ind_b],
                )

                v_pred = self.vf(observation_b)
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

                """ Value Update """
                self.value_optim.zero_grad()
                vf_loss.backward()
                self.value_optim.step()

                logp, entropy = self.policy.get_log_prob_entropy(
                    observation_b, action_b
                )
                ratio = torch.exp(logp - fixed_log_b)
                surr1 = ratio * advantages_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * advantages_b
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                """ Policy Update """
                self.policy_optim.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 20)
                self.policy_optim.step()

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
            self.eval_statistics["VF Loss"] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics["PG Loss"] = np.mean(ptu.get_numpy(pg_loss))

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
