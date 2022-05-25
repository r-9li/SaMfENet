import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from lib.ReaT_Net_connect import ResT
from lib.SK_SE_Net import SKConvBlock


class Upsample_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_Module, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class Encoder_Decoder_Net(nn.Module):
    def __init__(self):
        super(Encoder_Decoder_Net, self).__init__()
        self.encoder = ResT()

        self.x3_up1 = Upsample_Module(256, 256)
        self.x4_up1 = Upsample_Module(512, 512)
        self.x4_up2 = Upsample_Module(512, 512)

        self.psp = PSPModule(896, 1536, (1, 2, 3, 6))

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = Upsample_Module(1536, 768)
        self.up_2 = Upsample_Module(768, 256)
        self.up_3 = Upsample_Module(256, 128)

        self.drop_2 = nn.Dropout2d(p=0.15)

        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.LogSoftmax()
        )
        self.final_plus = SKConvBlock(in_channels=64, out_channels=256, num_branches=3)

    def forward(self, x):
        _, _, H, W = x.shape
        t_H = H // 8
        t_W = W // 8

        x2, x3, x4 = self.encoder(x)  # x2[B,128,H/8,W/8] x3[B,256,H/16,W/16]    x4[B,512,H/32,W/32]

        x3 = self.x3_up1(x3)  # 256,H/8,W/8

        x4 = self.x4_up1(x4)  # 512,H/16,W/16
        x4 = self.x4_up2(x4)  # 512,H/8,W/8

        x2 = F.upsample(input=x2, size=(t_H, t_W), mode='bilinear')
        x3 = F.upsample(input=x3, size=(t_H, t_W), mode='bilinear')  # 尺寸统一
        x4 = F.upsample(input=x4, size=(t_H, t_W), mode='bilinear')

        x_psp = torch.cat([x2, x3, x4], 1)  # 896,H/8,W/8

        p = self.psp(x_psp)  # 1536,H/8,W/8
        p = self.drop_1(p)

        p = self.up_1(p)  # 768,H/4,W/4
        p = self.drop_2(p)

        p = self.up_2(p)  # 256,H/2,W/2
        p = self.drop_2(p)

        p = self.up_3(p)  # 128,H,W

        p = self.final(p)  # 64,H,W
        p = self.final_plus(p)  # 256,H,W

        return p
