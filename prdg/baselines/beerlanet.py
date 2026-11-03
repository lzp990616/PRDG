"""BeerLaNet dagger: external official core + local detection adapter.

Set BEERLANET_SRC to the official src directory at the pinned commit.
The official core is not redistributed. See THIRD_PARTY_NOTICES.md.
"""
import hashlib, importlib.util, os, sys
from pathlib import Path
import torch
from torch import nn
def load_core():
    path = Path(os.environ['BEERLANET_SRC']) / 'BeerLaNet.py'
    expected = 'c22bb5294211eea63b3149e839b5610604a3ccc73eb155a631e8945663451900'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('BeerLaNet core differs from the paper commit')
    spec = importlib.util.spec_from_file_location('BeerLaNet', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['BeerLaNet'] = module
    spec.loader.exec_module(module)
    return module.BeerLaNet
class BeerLaNetDagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=load_core()(r=8,c=3,learn_S_init=True,calc_tau=True)
        self.adapt_conv=nn.Conv2d(8,3,1)
    def _apply(self,fn):
        super()._apply(fn)
        # Ultralytics saves EMA as half; retain this numerical front end in FP32.
        self.core.float();self.adapt_conv.float()
        return self
    def forward(self,x):
        # Core algebra is FP32, independent of detector AMP. No clipping/log/reconstruction added.
        with torch.autocast(device_type=x.device.type,enabled=False):
            _,_,density=self.core(x.float().contiguous(),S=None,D=None,n_iter=10,unit_norm_S=True)
            output=self.adapt_conv(density)
        return output.to(x.dtype)

class BeerInputStem(nn.Module):
    def __init__(self,block):
        super().__init__();self.beer=BeerLaNetDagger();self.block=block
        for k in ('i','f','type','np'):
            if hasattr(block,k):setattr(self,k,getattr(block,k))
    def forward(self,x):return self.block(self.beer(x))
