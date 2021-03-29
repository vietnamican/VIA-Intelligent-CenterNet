from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math

from .base import Base
from .layers import FPN

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

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CenterNet(Base):
    def __init__(self, base, heads,head_conv=128):
        super().__init__()
        self.heads = heads
        self.base = base
        channels = self.base.channels
        self.fpn = FPN(channels, out_dim=head_conv)
        for head in self.heads:
            classes = self.heads[head]
            fc =nn.Conv2d(head_conv, classes,
                          kernel_size=1, stride=1,
                          padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                nn.init.normal_(fc.weight, std=0.001)
                nn.init.constant_(fc.bias, 0)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.fpn(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]