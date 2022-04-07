from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer


class GCSL(Trainer):
    """
    Goal conditioned supervised learning

    https://github.com/dibyaghosh/gcsl
    https://arxiv.org/abs/1912.06088

    Using goal-conditioned value / policy function.
    """

    def __init__(
        self,
        policy,
        mode="MSE",
        reward_scale=1.0,
        discount=0.99,
        policy_lr=1e-3,
        optimizer_class=optim.Adam,
        use_horizons=False,
        **kwargs
    ):
        assert mode in ["MLE", "MSE", "CLASS"], "Invalid mode!"

        self.policy = policy
        self.mode = mode

        self.reward_scale = reward_scale
        self.discount = discount

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.eval_statistics = None
        self.classfication_criterion = nn.CrossEntropyLoss()

        self._n_train_steps_total = 0
        self.use_horizons = use_horizons

    def train_step(self, batch):

        obs = batch["observations"]
        actions = batch["actions"]
        horizons = batch["horizons"]
        goals = batch["desired_goals"]

        if self.use_horizons:
            concat_input = torch.cat([obs, goals, horizons], axis=-1)
        else:
            concat_input = torch.cat([obs, goals], axis=-1)

        if self.mode == "MLE":
            log_prob = self.policy.get_log_prob(concat_input, actions)
            policy_loss = -1.0 * log_prob.mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["Log-Likelihood"] = ptu.get_numpy(
                    -1.0 * policy_loss
                )
        elif self.mode == "MSE":
            pred_acts = self.policy(concat_input, deterministic=True)[0]
            squared_diff = (pred_acts - actions) ** 2
            policy_loss = torch.sum(squared_diff, dim=-1).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["MSE"] = ptu.get_numpy(policy_loss)
        elif self.mode == "CLASS":
            pred_acts, pred_act_logits, _ = self.policy(
                concat_input, deterministic=True
            )
            actions = actions.squeeze().type(torch.long)
            pred_acts = pred_acts.squeeze().type(torch.long)
            policy_loss = self.classfication_criterion(pred_act_logits, actions)
            accuracy = (pred_acts == actions).type(torch.FloatTensor).mean()

            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["CE Loss"] = ptu.get_numpy(policy_loss)
                self.eval_statistics["Accuracy"] = ptu.get_numpy(accuracy)
        else:
            raise NotImplementedError

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._n_train_steps_total += 1

    def get_snapshot(self):
        return dict(
            policy=self.policy,
        )

    @property
    def networks(self):
        return [self.policy]

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
