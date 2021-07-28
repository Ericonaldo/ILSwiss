from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.utils.normalizerpytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.algorithms.torch_rl_algorithm import (
    TorchRLAlgorithm,
    MetaTorchRLAlgorithm,
    NPMetaTorchRLAlgorithm,
)
from rlkit.torch.algorithms.sac.policies import MakeDeterministic


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class MetaSoftQLearning(MetaTorchRLAlgorithm):
    def __init__(
        self,
        env_sampler,
        policy,
        qf,
        discrete_policy=True,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        sql_one_over_alpha=1.0,
        soft_target_tau=1e-2,
        plotter=None,
        render_eval_paths=False,
        eval_deterministic=True,
        **kwargs
    ):
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        super().__init__(
            env_sampler=env_sampler,
            exploration_policy=policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.qf = qf
        self.policy = policy
        self.soft_target_tau = soft_target_tau
        # self.policy_mean_reg_weight = policy_mean_reg_weight
        # self.policy_std_reg_weight = policy_std_reg_weight
        # self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        # self.discrete_policy = discrete_policy

        self.sql_one_over_alpha = sql_one_over_alpha

        self.target_qf = qf.copy()
        self.eval_statistics = None

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        QF Loss
        """
        q_preds = self.qf(obs)
        target_q_next_obs_preds = self.target_qf(next_obs)
        target_v_next_obs_pred = (
            logsumexp(
                self.sql_one_over_alpha * target_q_next_obs_preds, dim=1, keepdim=True
            )
            / self.sql_one_over_alpha
        )
        q_target = rewards + (1.0 - terminals) * self.discount * target_v_next_obs_pred
        qf_loss = 0.5 * torch.mean((q_preds - q_target.detach()) ** 2)

        # print(q_preds[0].data.numpy())

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

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
            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Predictions",
                    ptu.get_numpy(q_preds),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(F.softmax(q_preds, dim=1)),
                )
            )

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf=self.qf,
            target_qf=self.target_qf,
        )
        return snapshot
