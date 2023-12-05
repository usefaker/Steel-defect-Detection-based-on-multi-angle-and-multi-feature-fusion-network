import torch
import math
import torch.nn as nn
from caffe2.python.layers.conv import Conv


class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut, baseWidth=26, scale=4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = Conv(inplanes, width * scale, k=1)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        for i in range(self.nums):
            convs.append(Conv(width, width, k=3))
        self.convs = nn.ModuleList(convs)

        self.conv3 = Conv(width * scale, planes * self.expansion, k=1, act=False)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width = width
        self.shortcut = shortcut

    def forward(self, x):

        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
                tp = sp
            sp = self.convs[i](sp)
            sp = sp+tp
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        out = self.silu(out)
        return out


