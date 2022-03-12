"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from typing import Callable, Dict, List, Tuple
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d

from rlkit.data_management.normalizer import FixedNormalizer
from rlkit.torch.utils import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
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
                ln = LayerNorm(next_size, **layer_norm_kwargs)
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

    @property
    def num_layers(self):
        return len(self.fcs) + 1


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class CatagorialMlp(Mlp):
    def __init__(
        self,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(**kwargs)

        self.softmax = nn.Softmax(dim=-1)

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
        output = self.softmax(self.output_activation(preactivation))
        # output = preactivation

        if return_preactivations:
            return output, preactivation
        else:
            return output


class EnsembleLinear(PyTorchModule):
    def __init__(self, input_size: int, output_size: int, ensemble_size: int = 1):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.zeros([self.ensemble_size, self.input_size, self.output_size])
        )
        self.bias = nn.Parameter(torch.zeros([self.ensemble_size, 1, self.output_size]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            output = torch.einsum("ij,ljk->lik", input, self.weight) + self.bias
        elif len(input.shape) == 3 and input.shape[0] == self.ensemble_size:
            output = torch.matmul(input, self.weight) + self.bias
        else:
            raise ValueError(
                f"In EnsembleLinear: invalid input dimension {input.shape} to layer shape {self.weight.shape}."
            )
        return output


class BNN(PyTorchModule):
    def __init__(
        self,
        hidden_sizes: List,
        output_size: int,
        input_size: int,
        init_w: float = 3e-3,
        hidden_activation: Callable = F.silu,
        output_activation: Callable = identity,
        hidden_init: Callable = ptu.fanin_init,
        b_init_value: float = 0.1,
        layer_norm: bool = False,
        layer_norm_kwargs: Dict = None,
        batch_norm: bool = False,
        batch_norm_before_output_activation: bool = False,
        num_nets: int = 1,
    ):
        self.save_init_params(locals())
        super().__init__()
        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        output_size *= 2
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_before_output_activation = batch_norm_before_output_activation
        self.num_nets = num_nets

        self.fcs = []
        self.layer_norms = []
        self.batch_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = EnsembleLinear(in_size, next_size, num_nets)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__(f"fc{i}", fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size, **layer_norm_kwargs)
                self.__setattr__(f"layer_norm{i}", ln)
                self.layer_norms.append(ln)
            if self.batch_norm:
                bn = BatchNorm1d(next_size)
                self.__setattr__(f"batch_norm{i}", bn)
                self.batch_norms.append(bn)

        if self.batch_norm_before_output_activation:
            bn = BatchNorm1d(output_size)
            self.__setattr__("batch_norm_last", bn)
            self.batch_norms.append(bn)

        self.last_fc = EnsembleLinear(in_size, output_size, num_nets)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.normalizer = FixedNormalizer(input_size)
        self.max_log_var = ptu.from_numpy(np.ones([1, output_size // 2]) / 2.0)
        self.min_log_var = ptu.from_numpy(-np.ones([1, output_size // 2]) * 10.0)

    def forward(
        self, input: torch.Tensor, ret_log_var: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = ptu.from_numpy(self.normalizer.normalize(input))
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
        mean = preactivation[:, :, : self.output_size // 2]
        mean = self.output_activation(mean)
        var = preactivation[:, :, self.output_size // 2 :]
        logvar = self.max_log_var - F.softplus(self.max_log_var - var)
        logvar = self.min_log_var + F.softplus(logvar - self.min_log_var)
        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def predict(
        self, input: torch.Tensor, factored: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_fac, std_fac = self.forward(input)
        if input.ndim == 2 and not factored:
            mean = torch.mean(mean_fac, dim=0)
            std = torch.mean(std_fac, dim=0) + torch.mean((mean_fac - mean) ** 2, dim=0)
            return mean, std
        else:
            return mean_fac, std_fac

    @property
    def num_layers(self) -> int:
        return len(self.fcs) + 1

    @property
    def layers(self) -> List[torch.nn.Module]:
        return self.fcs + [self.last_fc]

    def to(self, device):
        for net in self.layers:
            net.to(device)
