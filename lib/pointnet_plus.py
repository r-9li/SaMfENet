import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from time import time
import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNet_Plus(nn.Module):
    def __init__(self):
        super(PointNet_Plus, self).__init__()

        self.conv1_ps1 = torch.nn.Conv2d(3, 32, 1)
        self.conv2_ps1 = torch.nn.Conv2d(32, 64, 1)

        self.conv1_ps2 = torch.nn.Conv2d(3, 64, 1)
        self.conv2_ps2 = torch.nn.Conv2d(64, 128, 1)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)  # 64+64=128

        self.ap_ps1 = torch.nn.AvgPool1d(64)
        self.ap_ps2 = torch.nn.AvgPool1d(4)

    def forward(self, x):  # 1xNx3
        _, N, _ = x.shape
        x_feat = x.clone()
        x_feat = x_feat.transpose(2, 1).contiguous()  # [B,3,N]

        group_idx_1 = query_ball_point(0.45, 64, x, x)  # [B,N,64]
        grouped_x_1 = index_points(x, group_idx_1)  # [B,N,64,3]
        grouped_x_1 = grouped_x_1.permute(0, 3, 2, 1)  # [B,3,64,N]
        grouped_x_1 = F.relu(self.conv1_ps1(grouped_x_1))  # [B,32,64,N]
        grouped_x_1 = F.relu(self.conv2_ps1(grouped_x_1))  # [B,64,64,N]
        grouped_x_1 = torch.max(grouped_x_1, 2)[0]  # [B,64,N]

        x_feat = F.relu(self.conv1(x_feat))  # [B,64,N]
        x_feat = torch.cat([x_feat, grouped_x_1], 1)  # [B,128,N]
        x_feat = F.relu(self.conv2(x_feat))  # [B,256,N]

        group_idx_2 = query_ball_point(0.45, 128, x, x)  # [B,N,128]
        grouped_x_2 = index_points(x, group_idx_2)  # [B,N,128,3]
        grouped_x_2 = grouped_x_2.permute(0, 3, 2, 1)  # [B,3,128,N]
        grouped_x_2 = F.relu(self.conv1_ps2(grouped_x_2))  # [B,64,128,N]
        grouped_x_2 = F.relu(self.conv2_ps2(grouped_x_2))  # [B,128,128,N]
        grouped_x_2 = torch.max(grouped_x_2, 2)[0]  # [B,128,N]

        x_feat = torch.cat([x_feat, grouped_x_2], 1)  # [B,384,N]
        return x_feat
