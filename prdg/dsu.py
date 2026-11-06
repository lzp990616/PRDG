"""Independent implementation of domain shifts with uncertainty (DSU).

Compatibility conventions: unbiased spatial/batch variances, epsilon before
square roots, NumPy application gate, mean-noise before scale-noise. Unbiased estimates retain the experiment protocol.
"""
import numpy as np
import torch
from torch import nn


class DistributionUncertainty(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, factor=1.0):
        super().__init__()
        self.p = p
        self.eps = eps
        self.factor = factor

    def _sample_statistic(self, center):
        # Batch uncertainty and Gaussian reparameterization.
        spread = torch.sqrt(torch.var(center, dim=0, keepdim=True,
                                      unbiased=True) + self.eps)
        noise = torch.randn_like(center)
        return center + noise * self.factor * spread.expand_as(center)

    def forward(self, x):
        if not self.training:
            return x
        # Keep the gate before the singleton check, including RNG consumption.
        if np.random.random() > self.p:
            return x
        if x.shape[0] < 2:
            return x
        # Eqs. (1)-(2), with the experiment's unbiased variance convention.
        location = torch.mean(x, dim=(2, 3))
        scale = torch.sqrt(torch.var(x, dim=(2, 3), unbiased=True) + self.eps)
        sampled_location = self._sample_statistic(location)
        sampled_scale = self._sample_statistic(scale)
        shape = (x.shape[0], x.shape[1], 1, 1)
        standardized = (x - location.reshape(shape)) / scale.reshape(shape)
        # Eq. (7): restore independently sampled channel statistics.
        return standardized * sampled_scale.reshape(shape) + sampled_location.reshape(shape)


class ShallowDSU(nn.Module):
    """Preserve detector routing metadata while augmenting a block's output."""
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.style = DistributionUncertainty()
        for attribute in ('i', 'f', 'type', 'np'):
            if hasattr(block, attribute):
                setattr(self, attribute, getattr(block, attribute))

    def forward(self, x):
        features = self.block(x)
        return self.style(features)
