"""DSU algorithm from https://github.com/lixiaotong97/DSU/blob/main/dsu.py.

Official defaults p0.5, eps1e-6, factor1.0; unbiased spatial and batch variance.
"""
import numpy as np
import torch
from torch import nn


class DistributionUncertainty(nn.Module):
    def __init__(self,p=.5,eps=1e-6,factor=1.):
        super().__init__()
        self.p,self.eps,self.factor=p,eps,factor

    def forward(self,x):
        if not self.training or np.random.random()>self.p:
            return x
        if len(x)<2:
            # Unbiased batch variance is undefined for a singleton final batch.
            return x
        mean=x.mean((2,3))
        std=(x.var((2,3))+self.eps).sqrt()
        uncertainty_mean=(mean.var(0,keepdim=True)+self.eps).sqrt().expand_as(mean)
        uncertainty_std=(std.var(0,keepdim=True)+self.eps).sqrt().expand_as(std)
        beta=mean+torch.randn_like(mean)*self.factor*uncertainty_mean
        gamma=std+torch.randn_like(std)*self.factor*uncertainty_std
        shape=(len(x),x.shape[1],1,1)
        return (x-mean.reshape(shape))/std.reshape(shape)*gamma.reshape(shape)+beta.reshape(shape)


class ShallowDSU(nn.Module):
    def __init__(self,block):
        super().__init__()
        self.block=block
        self.style=DistributionUncertainty()
        for key in ('i','f','type','np'):
            if hasattr(block,key): setattr(self,key,getattr(block,key))

    def forward(self,x):
        return self.style(self.block(x))
