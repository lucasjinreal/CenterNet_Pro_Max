"""


centerface head:

this head takes MobilenetV2 as backbone, and outputs

hm, cls, landmark


"""

import torch
from torch import nn
import math

"""
fill up weights way same as CenterNet
but here we make it a separately function
"""


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeconvLayer(nn.Module):
    def __init__(self, out_dim, channel):
        super(DeconvLayer, self).__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                output_padding=0, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv2d(channel, out_dim,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True))
        # self.smooth = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])
        y = self.conv(layers[1])
        # out = self.smooth(x + y)
        out = x + y
        return out


class CenterFaceDeconv(nn.Module):
    def __init__(self, channels, out_dim=24):
        super(CenterFaceDeconv, self).__init__()
        channels = channels[::-1]
        self.conv = nn.Sequential(
            nn.Conv2d(channels[0], out_dim,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True))
        self.conv_last = nn.Sequential(
            nn.Conv2d(out_dim, out_dim,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.01),
            nn.ReLU(inplace=True))

        for i, channel in enumerate(channels[1:]):
            setattr(self, 'up_%d' % (i), DeconvLayer(out_dim, channel))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        x = self.conv(layers[-1])
        for i in range(0, len(layers) - 1):
            up = getattr(self, 'up_{}'.format(i))
            x = up([x, layers[len(layers) - 2 - i]])
        x = self.conv_last(x)
        return x


