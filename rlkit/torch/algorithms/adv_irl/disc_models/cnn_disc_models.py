import torch
import torch.nn as nn

from rlkit.torch.common.encoders import OUT_DIM, OUT_DIM_64


class CNNDisc(nn.Module):
    def __init__(
        self,
        input_shape,  # observation image
        input_dim=0,  # action
        num_filters=32,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act="relu",
        clamp_magnitude=10.0,
    ):
        super().__init__()

        if hid_act == "relu":
            hid_act_class = nn.ReLU
        elif hid_act == "tanh":
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude

        super().__init__()

        assert len(input_shape) == 3
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.num_layers = num_layer_blocks

        self.convs_list = nn.ModuleList(
            [nn.Conv2d(input_shape[0], num_filters, 3, stride=2)]
        )
        self.convs_list.append(hid_act_class())
        for i in range(self.num_layers - 1):
            self.convs_list.append(hid_act_class())
            self.convs_list.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.conv_model = nn.Sequential(*self.convs_list)

        out_dim = (
            OUT_DIM_64[self.num_layers]
            if input_shape[-1] == 64
            else OUT_DIM[self.num_layers]
        )
        self.mod_list = nn.ModuleList(
            [nn.Linear(num_filters * out_dim * out_dim + self.input_dim, hid_dim)]
        )
        self.mod_list.append(nn.LayerNorm(hid_dim))
        self.mod_list.append(hid_act_class())

        for i in range(self.num_layers - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            self.mod_list.append(nn.LayerNorm(hid_dim))
            self.mod_list.append(hid_act_class())

        self.mod_list.append(nn.Linear(hid_dim, 1))
        self.linear_model = nn.Sequential(*self.mod_list)

    def forward(self, obs, vec=None):
        output = self.conv_model(obs)
        output = output.view(output.size(0), -1)
        if self.input_dim != 0:
            assert vec is not None, "act should not be none!"
            output = torch.cat([output, vec], axis=-1)
        output = self.model(output)
        output = torch.clamp(
            output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude
        )
        return output


class ResNetCNNDisc(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act="relu",
        use_bn=True,
        clamp_magnitude=10.0,
    ):
        super().__init__()

        raise NotImplementedError
