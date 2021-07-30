import numpy as np
from numpy.random import choice

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.common.distributions import TanhNormal, ReparamTanhMultivariateNormal
from rlkit.torch.core import PyTorchModule


class DiscreteQWrapperPolicy(PyTorchModule, ExplorationPolicy):
    def __init__(self, qf):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        temp = self.eval_np(obs_np, deterministic=deterministic)[0]
        return temp

    def forward(self, obs, deterministic=False, return_log_prob=False):
        logits = self.qf(obs)
        log_probs = F.log_softmax(logits, dim=1)

        if deterministic:
            _, idx = torch.max(log_probs, 1)
            return (torch.unsqueeze(idx, 1), None)
        else:
            # Using Gumbel-Max trick to sample from the multinomials
            u = Variable(torch.rand(log_probs.size()))
            gumbel = -torch.log(-torch.log(u))
            _, idx = torch.max(gumbel + log_probs, 1)

            idx = torch.unsqueeze(idx, 1)
            log_prob = torch.gather(log_probs, 1, idx)

            return (idx, log_prob)
