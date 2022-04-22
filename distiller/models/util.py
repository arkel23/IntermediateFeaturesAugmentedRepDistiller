from __future__ import print_function

import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange

from .transformer import Transformer


class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.classifier(x)


class Rescaler(nn.Module):
    def __init__(self, opt, model, model_name):
        super().__init__()

        self.detach = opt.rs_detach

        if opt.rs_no_pool:
            original_dimensions, seq_lens = self.get_reduction_dims(
                model, opt.image_size, opt.cont_no_l, no_pool=opt.rs_no_pool)
        else:
            original_dimensions = self.get_reduction_dims(
                model, opt.image_size, opt.cont_no_l)
        final_dim = original_dimensions[-1]

        if model_name not in ['B_16', 'B_32', 'L_16']:
            if opt.rs_no_l_ada:
                no_l = [1]
                if opt.cont_no_l >= 2:
                    no_l.append(2)
                if opt.cont_no_l > 2:
                    while len(no_l) != opt.cont_no_l:
                        no_l.append(3)
                no_l.reverse()
                if opt.rs_mixer or opt.rs_transformer:
                    if opt.rs_mixer:
                        attn = 'mixer'
                    else:
                        attn = 'vanilla'

                    self.rescaling_head = nn.ModuleList([
                        nn.Sequential(
                            Transformer(
                                num_layers=l,
                                dim=original_dim,
                                num_heads=2,
                                ff_dim=int(original_dim * 2),
                                hidden_dropout_prob=opt.hidden_dropout_prob,
                                attention_probs_dropout_prob=opt.attention_probs_dropout_prob,
                                layer_norm_eps=opt.layer_norm_eps,
                                sd=opt.sd,
                                attn=attn,
                                seq_len=s),
                            Rearrange('b s c -> b c s'),
                            nn.AdaptiveAvgPool1d(1),
                            Rearrange('b c 1 -> b c'),
                            MLP(
                                layer_norm=opt.rs_ln, batch_norm=opt.rs_bn,
                                no_layers=1,
                                hidden_size=opt.rs_hid_dim,
                                in_features=original_dim,
                                out_features=final_dim, rescaler=True)
                        )
                        for original_dim, l, s in zip(original_dimensions, no_l, seq_lens)])
                else:
                    self.rescaling_head = nn.ModuleList([
                        MLP(
                            layer_norm=opt.rs_ln, batch_norm=opt.rs_bn,
                            no_layers=l, hidden_size=opt.rs_hid_dim,
                            in_features=original_dim, out_features=final_dim, rescaler=True)
                        for original_dim, l in zip(original_dimensions, no_l)])
            else:
                if opt.rs_mixer or opt.rs_transformer:
                    if opt.rs_mixer:
                        attn = 'mixer'
                    else:
                        attn = 'vanilla'
                    self.rescaling_head = nn.ModuleList([
                        nn.Sequential(
                            Transformer(
                                num_layers=opt.rs_no_l,
                                dim=original_dim,
                                num_heads=2,
                                ff_dim=int(original_dim * 2),
                                hidden_dropout_prob=opt.hidden_dropout_prob,
                                attention_probs_dropout_prob=opt.attention_probs_dropout_prob,
                                layer_norm_eps=opt.layer_norm_eps,
                                sd=opt.sd,
                                attn=attn,
                                seq_len=s),
                            Rearrange('b s c -> b c s'),
                            nn.AdaptiveAvgPool1d(1),
                            Rearrange('b c 1 -> b c'),
                            MLP(
                                layer_norm=opt.rs_ln, batch_norm=opt.rs_bn, no_layers=1,
                                hidden_size=opt.rs_hid_dim, in_features=original_dim,
                                out_features=final_dim, rescaler=True)
                        )
                        for original_dim, s in zip(original_dimensions, seq_lens)])
                else:
                    self.rescaling_head = nn.ModuleList([
                        MLP(
                            layer_norm=opt.rs_ln, batch_norm=opt.rs_bn,
                            no_layers=opt.rs_no_l, hidden_size=opt.rs_hid_dim,
                            in_features=original_dim, out_features=final_dim, rescaler=True)
                        for original_dim in original_dimensions])
        else:
            self.rescaling_head = nn.ModuleList([
                nn.Identity() for _ in original_dimensions])

    def get_reduction_dims(self, model, image_size, no_layers, no_pool=False):
        img = torch.rand(2, 3, image_size, image_size)
        out = model(img, classify_only=False)
        if no_pool:
            dims = [layer_output.size(2) for layer_output in out[:-1]]
            seq_lens = [layer_output.size(1) for layer_output in out[:-1]]
            return dims[-no_layers:], seq_lens[-no_layers:]
        else:
            dims = [layer_output.size(1) for layer_output in out[:-1]]
            return dims[-no_layers:]

    def forward(self, x):
        if self.detach:
            return [self.rescaling_head[i](features.detach()) for i, features in enumerate(x)]
        else:
            return [self.rescaling_head[i](features) for i, features in enumerate(x)]


class MLP(nn.Module):
    def __init__(self, linear: bool = False,
                 layer_norm: bool = False, batch_norm: bool = False,
                 no_layers: int = 3, in_features: int = None,
                 out_features: int = None, hidden_size: int = None,
                 rescaler: bool = False, proj_out_norm: bool = False,
                 layer_norm_eps: float = 1e-12, dropout_prob: float = 0.1):
        super().__init__()

        if no_layers != 1 and not hidden_size:
            hidden_size = out_features

        if linear:
            self.mlp = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False)
            )
        else:
            if no_layers == 1:
                self.mlp = nn.Sequential(
                    nn.Linear(in_features, out_features, bias=rescaler),
                )
            elif batch_norm:
                if no_layers == 2:
                    self.mlp = nn.Sequential(
                        nn.Linear(in_features, hidden_size, bias=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, out_features, bias=rescaler),
                    )
                else:
                    self.mlp = nn.Sequential(
                        nn.Linear(in_features, hidden_size, bias=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, hidden_size, bias=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, out_features, bias=rescaler),
                    )
            elif layer_norm:
                if no_layers == 2:
                    self.mlp = nn.Sequential(
                        nn.Linear(in_features, hidden_size, bias=True),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Linear(hidden_size, out_features, bias=rescaler),
                    )
                else:
                    self.mlp = nn.Sequential(
                        nn.Linear(in_features, hidden_size, bias=True),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Linear(hidden_size, hidden_size, bias=True),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Linear(hidden_size, out_features, bias=rescaler),
                    )
            elif not layer_norm and not batch_norm:
                if no_layers == 2:
                    self.mlp = nn.Sequential(
                        nn.Linear(in_features, hidden_size, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, out_features, bias=rescaler),
                    )
                else:
                    self.mlp = nn.Sequential(
                        nn.Linear(in_features, hidden_size, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, hidden_size, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, out_features, bias=rescaler),
                    )

            if rescaler:
                if batch_norm:
                    self.mlp = nn.Sequential(
                        self.mlp,
                        nn.BatchNorm1d(out_features),
                        nn.ReLU()
                    )
                elif layer_norm:
                    self.mlp = nn.Sequential(
                        self.mlp,
                        nn.LayerNorm(out_features, eps=layer_norm_eps),
                        nn.GELU()
                    )
                else:
                    self.mlp = nn.Sequential(
                        self.mlp,
                        nn.ReLU(),
                    )
            elif proj_out_norm:
                if batch_norm:
                    self.mlp = nn.Sequential(
                        self.mlp,
                        nn.BatchNorm1d(out_features),
                    )
                elif layer_norm:
                    self.mlp = nn.Sequential(
                        self.mlp,
                        nn.LayerNorm(out_features, eps=layer_norm_eps),
                    )

    def forward(self, x):
        return self.mlp(x)


class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""

    def __init__(self, t_shape, k=0.5, use_bn=False):
        super(Paraphraser, self).__init__()
        in_channel = t_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        factor = self.encoder(f_s)
        if is_factor:
            return factor
        rec = self.decoder(factor)
        return factor, rec


class Translator(nn.Module):
    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        super(Translator, self).__init__()
        in_channel = s_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        return self.encoder(f_s)


class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"""

    def __init__(self, s_shapes, t_shapes):
        super(Connector, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(
            self._make_conenctors(s_shapes, t_shapes))

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        connectors = []
        for s, t in zip(s_shapes, t_shapes):
            if s[1] == t[1] and s[2] == t[2]:
                connectors.append(nn.Sequential())
            else:
                connectors.append(ConvReg(s, t, use_relu=False))
        return connectors

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""

    def __init__(self, s_shapes, t_shapes):
        super(ConnectorV2, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(
            self._make_conenctors(s_shapes, t_shapes))

    def _make_conenctors(self, s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        t_channels = [t[1] for t in t_shapes]
        s_channels = [s[1] for s in s_shapes]
        connectors = nn.ModuleList([self._build_feature_connector(t, s)
                                    for t, s in zip(t_channels, s_channels)])
        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(
                s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented(
                'student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class Regress(nn.Module):
    """Simple Linear Regression for hints"""

    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""

    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module(
                    'MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module(
                    'AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(
            nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
