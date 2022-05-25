import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from time import time
import numpy as np
from lib.SK_SE_Net import Concurrent


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


class PointNet_Plus_Module(nn.Module):  # 128D
    def __init__(self, num_points):
        super(PointNet_Plus_Module, self).__init__()

        self.num_points = num_points

        self.conv1 = torch.nn.Conv2d(3, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1)

    def forward(self, x):
        group_idx = query_ball_point(0.6, self.num_points, x, x)  # [B,N,num]
        group_x = index_points(x, group_idx)  # [B,N,num,3]
        group_x = group_x.permute(0, 3, 2, 1)  # [B,3,num,N]
        group_x = F.relu(self.conv1(group_x))  # [B,64,num,N]
        group_x = F.relu(self.conv2(group_x))  # [B,128,num,N]
        group_x = torch.max(group_x, 2)[0]  # [B,128,N]
        return group_x


class SK_PointNet(nn.Module):
    def __init__(self, num_branches):
        super(SK_PointNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.num_branches = num_branches

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = torch.nn.Conv1d(128, 16, 1)
        self.fc2 = torch.nn.Conv1d(16, 128 * num_branches, 1)
        self.softmax = nn.Softmax(dim=1)

        self.branches = Concurrent(stack=True)
        stride = 16
        for i in range(num_branches):
            stride = stride * 2
            self.branches.add_module("branch{}".format(i), PointNet_Plus_Module(num_points=stride))

    def forward(self, x):  # [B,N,3]
        y = self.branches(x)  # [B,branches,128,N]

        u = y.sum(dim=1)  # [B,128,N]
        s = self.pool(u)  # [B,128,1]
        z = F.relu(self.fc1(s))  # [B,16,1]
        w = self.fc2(z)  # [B,128*branches,1]

        batch = w.size(0)
        w = w.view(batch, self.num_branches, 128)  # [B,branches,128]
        w = self.softmax(w)  # [B,branches,128]
        w = w.unsqueeze(-1)  # [B,branches,128,1]

        y = y * w  # [B,branches,128,N]
        y = y.sum(dim=1)  # [B,128,N]

        x_single = x.clone()
        x_single = x_single.transpose(2, 1).contiguous()  # [B,3,N]
        x_single = F.relu(self.conv1(x_single))  # [B,64,N]
        x_single = F.relu(self.conv2(x_single))  # [B,128,N]
        x_single = F.relu(self.conv3(x_single))  # [B,256,N]

        x_feat = torch.cat([x_single, y], 1)  # 256+128=384

        return x_feat  # [B,384,N]
