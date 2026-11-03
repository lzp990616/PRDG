"""Training-template-only Reinhard / Macenko / Vahadane."""
from .vahadane import vahadane_template, vahadane_transform
import cv2,numpy as np
def macenko_components(bgr):
    rgb=bgr[:,:,::-1].reshape(-1,3).astype(np.float64)
    od=-np.log((rgb+1.)/240.)
    tissue=od[(od>=.15).all(1)]
    if len(tissue)<8:raise ValueError('Macenko: fewer than8 optically dense pixels')
    _,vectors=np.linalg.eigh(np.cov(tissue.T));plane=vectors[:,1:3]
    angles=np.arctan2(*(tissue@plane)[:,::-1].T)
    limits=np.percentile(angles,[1,99])
    basis=plane@np.stack([np.cos(limits),np.sin(limits)])
    basis=basis[:,np.argsort(basis[0])[::-1]]
    if np.linalg.matrix_rank(basis)!=2:raise ValueError('Macenko: rank-deficient stain basis')
    concentrations=np.linalg.lstsq(basis,od.T,rcond=None)[0]
    scale=np.percentile(concentrations,99,axis=1)
    if not np.isfinite(scale).all() or (scale<=1e-8).any():raise ValueError('Macenko: invalid p99 concentrations')
    return basis,concentrations,scale
def fit(bgr,method):
    if method=='Vahadane':return vahadane_template(bgr)
    if method=='Macenko':
        b,_,s=macenko_components(bgr);return dict(basis=b.tolist(),scale=s.tolist())
    if method=='Reinhard':
        lab=cv2.cvtColor(bgr.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)
        return dict(mean=lab.mean((0,1)).tolist(),std=lab.std((0,1)).tolist())
    raise ValueError(method)
def transform(bgr,method,target):
    if method=='Vahadane':return vahadane_transform(bgr,target)
    if method=='Macenko':
        _,c,s=macenko_components(bgr)
        density=np.asarray(target['basis'])@(c*(np.asarray(target['scale'])/s)[:,None])
        # Clip exponent only outside the representable uint8 range.
        rgb=240.*np.exp(np.clip(-density,-80,np.log(255./240.)))
        out=np.clip(rgb.T.reshape(bgr.shape),0,255).astype(np.uint8)[:,:,::-1].copy()
    elif method=='Reinhard':
        lab=cv2.cvtColor(bgr.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)
        z=(lab-lab.mean((0,1)))/np.maximum(lab.std((0,1)),1e-6)
        lab=(z*np.asarray(target['std'])+np.asarray(target['mean'])).astype(np.float32)
        out=np.rint(np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),0,1)*255).astype(np.uint8)
    else:raise ValueError(method)
    if not np.isfinite(out).all():raise ValueError('Nonfinite stain output')
    return out
