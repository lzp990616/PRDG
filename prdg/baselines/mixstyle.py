"""MixStyle algorithm adapted from Kaiyang Zhou's official Dassl implementation.

Reference: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/ops/mixstyle.py
Random mixing, detached instance statistics, unbiased spatial variance, Beta(alpha,alpha).
"""
import random
import torch
from torch import nn


class MixStyle(nn.Module):
    def __init__(self, p=.5, alpha=.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        mean = x.mean((2, 3), keepdim=True).detach()
        std = (x.var((2, 3), keepdim=True) + self.eps).sqrt().detach()
        coefficient = torch.distributions.Beta(self.alpha, self.alpha).sample((len(x), 1, 1, 1)).to(x.device)
        order = torch.randperm(len(x))
        mixed_mean = coefficient * mean + (1 - coefficient) * mean[order]
        mixed_std = coefficient * std + (1 - coefficient) * std[order]
        return (x - mean) / std * mixed_std + mixed_mean


class ShallowMixStyle(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.style = MixStyle(p=.5, alpha=.1)
        for key in ('i', 'f', 'type', 'np'):
            if hasattr(block, key):
                setattr(self, key, getattr(block, key))

    def forward(self, x):
        return self.style(self.block(x))
