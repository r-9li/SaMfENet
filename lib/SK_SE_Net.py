import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        mid_channels = channels // 8
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=mid_channels, kernel_size=1, stride=1, groups=1,
                               bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=mid_channels, out_channels=channels, kernel_size=1, stride=1, groups=1,
                               bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)

        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)

        w = self.sigmoid(w)
        x = x * w
        return x


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=False,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):

    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=False,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):

    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class Concurrent(nn.Sequential):

    def __init__(self,
                 axis=1,
                 stack=False,
                 merge_type=None):
        super(Concurrent, self).__init__()
        assert (merge_type is None) or (merge_type in ["cat", "stack", "sum"])
        self.axis = axis
        if merge_type is not None:
            self.merge_type = merge_type
        else:
            self.merge_type = "stack" if stack else "cat"

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.merge_type == "stack":
            out = torch.stack(tuple(out), dim=self.axis)
        elif self.merge_type == "cat":
            out = torch.cat(tuple(out), dim=self.axis)
        elif self.merge_type == "sum":
            out = torch.stack(tuple(out), dim=self.axis).sum(self.axis)
        else:
            raise NotImplementedError()
        return out


class SKConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=32,
                 num_branches=2,
                 reduction=16,
                 min_channels=32):
        super(SKConvBlock, self).__init__()
        self.num_branches = num_branches
        self.out_channels = out_channels
        mid_channels = max(in_channels // reduction, min_channels)

        self.branches = Concurrent(stack=True)
        for i in range(num_branches):
            dilation = 1 + i
            self.branches.add_module("branch{}".format(i + 2), conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=groups))
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = conv1x1_block(
            in_channels=out_channels,
            out_channels=mid_channels)
        self.fc2 = nn.Conv2d(in_channels=mid_channels, out_channels=(out_channels * num_branches), kernel_size=1,
                             stride=1, groups=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        y = self.branches(x)

        u = y.sum(dim=1)
        s = self.pool(u)
        z = self.fc1(s)
        w = self.fc2(z)

        batch = w.size(0)
        w = w.view(batch, self.num_branches, self.out_channels)
        w = self.softmax(w)
        w = w.unsqueeze(-1).unsqueeze(-1)

        y = y * w
        y = y.sum(dim=1)
        return y
