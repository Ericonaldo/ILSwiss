"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import math
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d

from rlkit.policies.base import Policy
from rlkit.torch.utils import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.utils.normalizer import TorchFixedNormalizer
from rlkit.torch.common.modules import LayerNorm


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
        batch_norm=False,
        # batch_norm_kwargs=None,
        batch_norm_before_output_activation=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_before_output_activation = batch_norm_before_output_activation
        self.fcs = []
        self.layer_norms = []
        self.batch_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

            if self.batch_norm:
                bn = BatchNorm1d(next_size)
                self.__setattr__("batch_norm{}".format(i), bn)
                self.batch_norms.append(bn)

        if self.batch_norm_before_output_activation:
            bn = BatchNorm1d(output_size)
            self.__setattr__("batch_norm_last", bn)
            self.batch_norms.append(bn)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        if self.batch_norm_before_output_activation:
            preactivation = self.batch_norms[-1](preactivation)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)