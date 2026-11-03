import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, shutil
import numpy as np
from prdg.io import rows, read_image, image_path, label_path, write_yaml, save
from prdg.normalization import fit_target, normalize
from prdg.baselines.references import transform as reference_transform
from prdg.baselines.stains import fit as stain_fit, transform as stain_transform

def prepare(fold,out,method='PRN'):
    fold=Path(fold).resolve();out=Path(out).resolve()
    if out==fold or fold in out.parents:raise ValueError('Output must not be inside input fold')
    records=rows(fold/'manifest.csv');training=[r for r in records if r['loso_split']=='train']
    held={r['source'] for r in records if r['loso_split']=='test'}
    assert not held & {r['source'] for r in training}
    out.mkdir(parents=True,exist_ok=False)
    target=fit_target(read_image(image_path(fold,r)) for r in training)
    template=sorted(training,key=lambda r:(r['source'],r['new_name']))[0]
    stains=stain_fit(read_image(image_path(fold,template)),method) if method in ('Reinhard','Macenko','Vahadane') else None
    ratio=None
    if method=='B3':
        values=[]
        for r in training:
            im=read_image(image_path(fold,r));h,w=im.shape[:2]
            for line in label_path(fold,r).read_text().splitlines():
                a=line.split()
                if a:values.append(float(np.hypot(float(a[3])*w,float(a[4])*h)/min(h,w)))
        ratio=max(values)
    import cv2
    for r in records:
        im=read_image(image_path(fold,r))
        if method=='PRN':im=normalize(im,target)
        elif stains is not None:im=stain_transform(im,method,stains)
        else:im=reference_transform(im,method,target,ratio)[0]
        dest=image_path(out,r);dest.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(dest),im):raise IOError(dest)
        lab=label_path(out,r);lab.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(label_path(fold,r),lab)
    shutil.copyfile(fold/'manifest.csv',out/'manifest.csv');write_yaml(out)
    save(out/'normalization.json',dict(method=method,target_rgb=target.tolist(),training_images=len(training),
        template=template['new_name'] if stains else None,template_source=template['source'] if stains else None,
        stain_target=stains,kernel_ratio=ratio,target_fitted_on='train only'))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--method',default='PRN',choices=['PRN','B0','B1','B2','B3','B4','B5','B6','Reinhard','Macenko','Vahadane'])
    a=ap.parse_args();prepare(a.fold,a.out,a.method)
if __name__=='__main__':main()
