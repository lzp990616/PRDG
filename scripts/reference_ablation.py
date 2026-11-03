import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import cv2, numpy as np, pandas as pd
from scipy.stats import pearsonr
from prdg.io import rows,read_image,image_path,label_path,write_csv
from prdg.baselines.references import reference
def coverage_mask(image,label):
    h,w=image.shape[:2];mask=np.zeros((h,w),bool)
    for line in Path(label).read_text().splitlines():
        a=line.split()
        if len(a)<5:continue
        _,x,y,bw,bh=map(float,a[:5]);x1=max(0,int((x-bw/2)*w));x2=min(w,int(np.ceil((x+bw/2)*w)))
        y1=max(0,int((y-bh/2)*h));y2=min(h,int(np.ceil((y+bh/2)*h)))
        mask[y1:y2,x1:x2]=True
    return mask
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--methods',nargs='+',default=['B0','B1','B2','B4','B5','B6']);ap.add_argument('--kernel-ratio',type=float)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);result=[]
    for row in rows(a.fold/'manifest.csv'):
        image=read_image(image_path(a.fold,row));coverage=coverage_mask(image,label_path(a.fold,row))
        for method in a.methods:
            ref,support,base,meta=reference(image,method,a.kernel_ratio)
            mixed=None if support is None else float((support & (coverage[:,:,None] if support.ndim==3 else coverage)).sum()/max(1,support.sum()))
            # Original diagnostic: mean relative error over channels (and pixels for B4).
            error=np.abs(ref-base)/np.maximum(base,1.)
            if error.ndim==3:error=error.mean((0,1))
            result.append(dict(source=row['source'],image=row['new_name'],method=method,rho=mixed,
                relative_error=float(np.mean(error)),coverage=float(coverage.mean())))
    write_csv(a.out/'references.csv',result);df=pd.DataFrame(result);corr=[]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(9,7))
    colors={'PBC':'#4C72B0','BCCD':'#DD8452','Raabin-WBC':'#55A868','BCDD':'#8172B3'}
    for ax,(source,g) in zip(axs.flat,df[df.method=='B5'].groupby('source')):
        r,p=pearsonr(g.coverage,g.relative_error);corr.append(dict(source=source,n=len(g),r=float(r),p=float(p)))
        ax.scatter(g.coverage,g.relative_error,s=6,color=colors[source]);ax.set(title=f'{source}: r={r:.3f}, p={p:.3g}',
            xlabel='Cell coverage (fraction of image in annotated boxes)',ylabel='Relative error of grey-world reference')
    write_csv(a.out/'correlations.csv',corr);fig.tight_layout();fig.savefig(a.out/'coverage.pdf');fig.savefig(a.out/'coverage.png',dpi=300)
if __name__=='__main__':main()

