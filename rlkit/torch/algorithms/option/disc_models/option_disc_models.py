import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.utils import pytorch_util as ptu
from rlkit.torch.algorithms.adv_irl.disc_models.simple_disc_models import MLPDisc


class OptionDisc(MLPDisc):
    def __init__(
        self,
        input_dim,
        option_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act="relu",
        use_bn=True,
        clamp_magnitude=10.0,
        use_option=True,
    ):
        super().__init__(
            input_dim=input_dim,
            num_layer_blocks=num_layer_blocks,
            hid_dim=hid_dim,
            hid_act=hid_act,
            use_bn=use_bn,
            clamp_magnitude=clamp_magnitude,
        )

        self.use_option = use_option
        self.option_dim = option_dim
        if self.use_option:
            self.mod_list[-1] = nn.Linear(
                hid_dim, ((self.option_dim + 1) * self.option_dim)
            )
        self.model = nn.Sequential(*self.mod_list)

    def forward(self, inputs, ct_1, ct):
        output = self.model(inputs)
        output = torch.clamp(
            output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude
        )
        if self.use_option:
            output = output.view(-1, self.option_dim + 1, self.option_dim)
            ct_1 = ct_1.view(-1, 1, 1).expand(-1, 1, self.option_dim)
            output = (
                output.gather(dim=-2, index=ct_1.type(torch.long))
                .squeeze(dim=-2)
                .gather(dim=-1, index=ct.type(torch.long))
            )
        return output
