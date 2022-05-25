import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False, num_group=32, first=False):
        super(Bottleneck, self).__init__()
        self.first = first
        stride = 1
        self.process_res = None
        if downsample:
            stride = 2
        if first:
            self.process_res = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.GroupNorm(num_groups=8, num_channels=out_channel),
            )
        self.conv1 = nn.Conv2d(in_channel, out_channel // 2, kernel_size=1, bias=False)
        self.bn1 = torch.nn.GroupNorm(num_groups=8, num_channels=out_channel // 2)
        self.conv2 = nn.Conv2d(out_channel // 2, out_channel // 2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = torch.nn.GroupNorm(num_groups=8, num_channels=out_channel // 2)
        self.conv3 = nn.Conv2d(out_channel // 2, out_channel, kernel_size=1, bias=False)
        self.bn3 = torch.nn.GroupNorm(num_groups=8, num_channels=out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.first:
            residual = self.process_res(x)

        out += residual
        out = self.relu(out)

        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            Bottleneck(in_channel=in_channels, out_channel=out_channels, downsample=True, first=True),
            Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
            Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
            Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
            Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                Bottleneck(in_channel=in_channels, out_channel=out_channels, downsample=False, first=True),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                Bottleneck(in_channel=in_channels, out_channel=out_channels, downsample=False, first=True),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False),
                Bottleneck(in_channel=out_channels, out_channel=out_channels, downsample=False, first=False))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_ResNext(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet_ResNext, self).__init__()
        self.bilinear = bilinear

        self.inc = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.inc_bn = torch.nn.GroupNorm(num_groups=8, num_channels=64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.relu = nn.ReLU(inplace=True)

        self.edge_bn = torch.nn.GroupNorm(num_groups=8, num_channels=32)
        self.edge_conv1 = nn.Conv2d(64, 32, kernel_size=1)
        self.edge_conv2 = nn.Conv2d(32, 1, kernel_size=1)

        self.bn = torch.nn.GroupNorm(num_groups=8, num_channels=128)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.inc_bn(x1)
        x1 = self.relu(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_edge = self.edge_conv1(x)
        x_edge = self.edge_bn(x_edge)
        x_edge = self.relu(x_edge)
        x_edge = self.edge_conv2(x_edge)  # 1,H,W

        x_out = self.conv1(x)
        x_out = self.bn(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv2(x_out)  # 256,H,W

        return x_edge, x_out
