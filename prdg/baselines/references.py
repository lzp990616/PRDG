"""Fixed reference ablations B0--B6 (all paired with DSU in the paper)."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from prdg.eps import background_mask_v2

def reference(bgr,method,kernel_ratio=None):
    rgb=bgr[:,:,::-1].astype(np.float32);h,w=rgb.shape[:2]
    hand=background_mask_v2(bgr)
    base=rgb[hand].mean(0)
    meta={'fallback':False,'kernel':0,'sigma':0.0}
    support=None
    if method=='B0':
        support=hand;ref=base
    elif method=='B1':
        v=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)[:,:,2]
        threshold,_=cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        support=v>threshold
        if not support.any():support=np.ones((h,w),bool);meta['fallback']=True
        ref=rgb[support].mean(0);meta['otsu']=float(threshold)
    elif method=='B2':
        lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
        if np.max(lab.max(0)-lab.min(0))==0:
            support=np.ones((h,w),bool);meta['fallback']=True
        else:
            cv2.setRNGSeed(42)
            _,labels,centers=cv2.kmeans(lab,2,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,.01),1,cv2.KMEANS_PP_CENTERS)
            support=(labels.ravel()==int(np.argmax(centers[:,0]))).reshape(h,w)
        ref=rgb[support].mean(0)
    elif method=='B3':
        if kernel_ratio is None:raise ValueError('B3 requires fixed training-fold kernel_ratio')
        k=int(np.floor(min(h,w)*kernel_ratio))+1;k+=1-k%2
        small=cv2.resize(rgb,(max(1,int(np.ceil(w/4))),max(1,int(np.ceil(h/4)))),interpolation=cv2.INTER_AREA)
        sk=int(np.ceil(k/4));sk+=1-sk%2
        bg=cv2.morphologyEx(small,cv2.MORPH_CLOSE,np.ones((sk,sk),np.uint8),borderType=cv2.BORDER_REFLECT_101)
        ref=bg.mean((0,1));meta['kernel']=k;meta['downsample_kernel']=sk;meta['kernel_ge_short_side']=k>=min(h,w)
    elif method=='B4':
        support=hand;sigma=max(1.,min(h,w)/8);meta['sigma']=sigma
        den=gaussian_filter(hand.astype(np.float32),sigma,mode='reflect',truncate=3.)
        ref=np.stack([gaussian_filter(rgb[:,:,c]*hand,sigma,mode='reflect',truncate=3.)/np.maximum(den,1e-8) for c in range(3)],axis=-1)
        low=den<1e-6;ref[low]=base;meta['fallback']=bool(low.any());meta['fallback_pixels']=int(low.sum())
    elif method=='B5':
        support=np.ones((h,w),bool);ref=rgb.mean((0,1))
    elif method=='B6':
        ref=np.percentile(rgb,99,axis=(0,1)).astype(np.float32)
        support=rgb>=ref[None,None,:]
    else:raise ValueError(method)
    assert np.isfinite(ref).all()
    return ref,support,base,meta

def transform(bgr,method,target,kernel_ratio=None):
    ref,support,base,meta=reference(bgr,method,kernel_ratio)
    rgb=bgr[:,:,::-1].astype(np.float32)
    gain=np.clip(np.asarray(target,np.float32)/np.maximum(ref,1.),.5,2.)
    y=np.rint(np.clip(rgb*gain,0,255)).astype(np.uint8)[:,:,::-1]
    return y,ref,support,base,meta
