import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import cv2, numpy as np, pandas as pd
from prdg.io import rows, read_image, image_path, write_csv
from prdg.eps import background_mask, background_mask_v2
from prdg.normalization import masked_mean
PARAMS={'dilation':[15,23,31,41,51],'dark_percentile':[10,15,20,25],'value_percentile':[40,50,60],'saturation_multiplier':[.8,.9,1.,1.1,1.2],'fallback_percent':[1,3,5]}
DEFAULTS=dict(dilation=31,dark_percentile=15,value_percentile=50,saturation_multiplier=1.,fallback_percent=3)
def variant(image,**changes):
    cfg={**DEFAULTS,**changes};hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);s=hsv[:,:,1];v=hsv[:,:,2]
    otsu=cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(cfg['dilation'],cfg['dilation']))
    excluded=cv2.dilate((v<np.quantile(v,cfg['dark_percentile']/100)).astype(np.uint8),kernel).astype(bool)
    candidate=(s<=otsu*cfg['saturation_multiplier'])&(v>=np.quantile(v,cfg['value_percentile']/100))&~excluded
    fallback=bool(candidate.mean()<cfg['fallback_percent']/100)
    return (background_mask(image) if fallback else candidate),fallback
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=Path,required=True,help='One prepared fold contains all 1260 images exactly once')
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    result=[];defaults=[]
    for r in rows(a.fold/'manifest.csv'):
        im=read_image(image_path(a.fold,r));base,audit=background_mask_v2(im,True);ref=masked_mean(im,base)
        defaults.append(dict(source=r['source'],image=r['new_name'],fallback=audit['fallback']))
        for param,values in PARAMS.items():
            for value in values:
                mask,fallback=variant(im,**{param:value});b=masked_mean(im,mask)
                if value==DEFAULTS[param]:assert np.array_equal(mask,base)
                result.append(dict(source=r['source'],image=r['new_name'],parameter=param,value=value,fallback=fallback,
                    mask_iou=float((mask&base).sum()/max(1,(mask|base).sum())),mask_fraction=float(mask.mean()),
                    reference_relative_difference=float(np.max(np.abs(b-ref)/np.maximum(np.abs(ref),1e-12)))))
    write_csv(a.out/'per_image.csv',result);write_csv(a.out/'defaults.csv',defaults)
    df=pd.DataFrame(result);summary=[]
    for (source,param,value),g in df.groupby(['source','parameter','value'],sort=False):
        row=dict(source=source,parameter=param,value=value,n=len(g),fallback_rate=g.fallback.mean())
        for k in ('mask_iou','mask_fraction','reference_relative_difference'):
            row.update({k+'_mean':g[k].mean(),k+'_p05':g[k].quantile(.05),k+'_p95':g[k].quantile(.95)})
        summary.append(row)
    write_csv(a.out/'summary.csv',summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6));data=pd.DataFrame(summary)
    colors={'PBC':'#4C72B0','BCCD':'#DD8452','Raabin-WBC':'#55A868','BCDD':'#8172B3'}
    for ax,param in zip(axes.flat,PARAMS):
        for source,g in data[data.parameter==param].groupby('source'):
            ax.plot(g.value,g.reference_relative_difference_mean,label=source,color=colors[source])
        ax.set(xlabel=param,ylabel='Relative reference difference')
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels())
    fig.tight_layout();fig.savefig(a.out/'sensitivity.pdf');fig.savefig(a.out/'sensitivity.png',dpi=300)
if __name__=='__main__':main()
