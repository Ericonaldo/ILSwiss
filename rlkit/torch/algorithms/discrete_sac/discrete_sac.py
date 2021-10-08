from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class DiscreteSoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    TODO: Recently in rlkit there is a version which only uses two Q functions
    as well as an implementation of entropy tuning but I have not implemented
    those
    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        reward_scale=1.0,
        discount=0.99,
        alpha=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        soft_target_tau=1e-2,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        **kwargs,
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()
        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )

        self.alpha = alpha

    def train_step(self, batch):
        # q_params = itertools.chain(self.qf1.parameters(), self.qf2.parameters())
        # v_params = itertools.chain(self.vf.parameters())
        # policy_params = itertools.chain(self.policy.parameters())

        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        actions = actions.long()

        """
        QF Loss
        """
        # Only unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.vf.parameters():
        #     p.requires_grad = False
        # for p in self.policy.parameters():
        #     p.requires_grad = False
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs).gather(1, actions)
        q2_pred = self.qf2(obs).gather(1, actions)
        _, next_action_log_prob = self.policy(next_obs)
        next_action_prob = next_action_log_prob.exp()
        next_action_dist = Categorical(probs=next_action_prob)
        next_q = next_action_dist.probs * torch.min(
            self.target_qf1(next_obs),
            self.target_qf2(next_obs),
        )
        target_v_values = next_q.sum(dim=-1) + self.alpha * next_action_dist.entropy()
        q_target = rewards + (
            1.0 - terminals
        ) * self.discount * target_v_values.unsqueeze(
            -1
        )  # original implementation has detach
        q_target = q_target.detach()
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        qf1_loss.backward()
        qf2_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        """
        Policy Loss
        """
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = False
        # for p in self.vf.parameters():
        #     p.requires_grad = False
        # for p in self.policy.parameters():
        #     p.requires_grad = True
        # new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        # q1_new_acts = self.qf1(obs, new_actions)
        # q2_new_acts = self.qf2(obs, new_actions)  # error
        # q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        _, action_log_prob = self.policy(obs)
        action_prob = action_log_prob.exp()
        action_dist = Categorical(probs=action_prob)
        current_q = torch.min(
            self.qf1(obs),
            self.qf2(obs),
        )
        current_q = current_q.detach()

        self.policy_optimizer.zero_grad()
        policy_loss = -torch.mean(
            self.alpha * action_dist.entropy()
            + (action_dist.probs * current_q).sum(dim=-1)
        )
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update networks
        """
        # unfreeze all -> initial states
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.vf.parameters():
        #     p.requires_grad = True
        # for p in self.policy.parameters():
        #     p.requires_grad = True

        # unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True

        self._update_target_network()

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

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
