'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2024-07-15 21:12:25
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2024-07-15 21:13:35
'''
import torch
from torch import nn

class RBF(nn.Module):

    def __init__(self, device,n_kernels=256, mul_factor=2.0, bandwidth=None):
        super().__init__()
        v1 = torch.arange(n_kernels)
        v3 = v1 - n_kernels // 2
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        self.bandwidth_multipliers=self.bandwidth_multipliers.to(X.device)
        L2_distances = torch.cdist(X, X) ** 2
        v1 = -L2_distances[None, ...]  # 1 4 256 48 48
        v2 = self.get_bandwidth(L2_distances)
        v3 = self.bandwidth_multipliers  # 5
        v4 = (v2 * v3)[:, None, None]    # 5 1 1
        v5 = torch.exp(v1 / v4).sum(dim=0)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, n_kernels,mul_factor=2.0):
        super().__init__()
        kernel=RBF('cuda',n_kernels,mul_factor)
        self.kernel = kernel

    def forward(self, X, Y):
        self.kernel = self.kernel.to(X.device)
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY