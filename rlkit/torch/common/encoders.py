import torch
import torch.nn as nn
from rlkit.torch.core import PyTorchModule


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}
# for 108 x 108 inputs
OUT_DIM_108 = {4: 47}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )


def make_decoder(decoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )


class PixelEncoder(PyTorchModule):
    """Convolutional encoder of pixels observations."""

    def __init__(
        self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False
    ):
        self.save_init_params(locals())
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]
        self.out_dim = out_dim
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        if obs.max() > 1.0:
            obs = obs / 255.0
        self.outputs["obs"] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs["tanh"] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class IdentityEncoder(PyTorchModule):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class PixelDecoder(PyTorchModule):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        self.save_init_params(locals())
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]
        self.out_dim = out_dim

        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1)
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs["fc"] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs["deconv1"] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs["deconv%s" % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs["obs"] = obs

        return obs


_AVAILABLE_ENCODERS = {"pixel": PixelEncoder, "identity": IdentityEncoder}
_AVAILABLE_DECODERS = {"pixel": PixelDecoder}
