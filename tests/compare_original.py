import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse,importlib.util,json,hashlib,ast,random
import cv2,numpy as np,torch
from prdg.eps import background_mask_v2
from prdg.normalization import channel_normalize,fit_target
from prdg.dsu import DistributionUncertainty
from prdg.baselines.mixstyle import MixStyle
from prdg.io import save
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    sys.dont_write_bytecode=True
    ap=argparse.ArgumentParser();ap.add_argument('--original',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();original=module(a.original/'code/train/image_color_normalizers.py','original_normalizers')
    import csv
    with (a.original/'results/paper_tables/dataset_images.csv').open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f))
    selected=[]
    for domain in ['PBC','Raabin','BCCD','BCDD']:
        selected+=sorted([r for r in records if r['source']==domain],key=lambda r:r['image'])[:5]
    target=np.array([203.7224,185.998,174.8505],np.float32);checks=[];ims=[]
    for row in selected:
        im=cv2.imread(row['image_path']);assert im is not None;ims.append(im)
        m0=original.background_mask_v2(im);m1=background_mask_v2(im)
        y0=original.channel_normalize(im,target,m0);y1=channel_normalize(im,target,m1)
        difference=int(np.abs(y0.astype(int)-y1.astype(int)).max())
        assert np.array_equal(m0,m1) and difference==0
        checks.append(dict(source=row['source'],image=row['image'],image_sha256=hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest(),mask_equal=True,max_abs_diff=difference))
    expected=np.asarray(np.mean([im[:,:,::-1][original.background_mask_v2(im)].mean(0) for im in ims],axis=0).astype(float).tolist(),np.float32)
    assert np.array_equal(expected,fit_target(ims))
    # Extra degenerate-input checks exercise fallback and rounding.
    for value in (0,127,255):
        image=np.full((32,32,3),value,np.uint8)
        assert np.array_equal(background_mask_v2(image),original.background_mask_v2(image))
    old_dsu=module(a.original/'code/models/dsu_shallow.py','original_dsu').DistributionUncertainty
    old_mix=module(a.original/'code/models/mixstyle_shallow.py','original_mix').MixStyle
    x=torch.randn(4,8,16,16);module_checks=[]
    for name,old,new in [('DSU',old_dsu,DistributionUncertainty),('MixStyle',old_mix,MixStyle)]:
        for training in (True,False):
            torch.manual_seed(42);np.random.seed(1);random.seed(1);y0=old().train(training)(x.clone())
            torch.manual_seed(42);np.random.seed(1);random.seed(1);y1=new().train(training)(x.clone())
            assert torch.equal(y0,y1);module_checks.append(dict(module=name,training=training,max_abs_diff=0))
    save(a.out,dict(passed=True,real_images=20,checks=checks,target_reduction_exact=True,module_checks=module_checks))
    print('PASS: 20 images, EPS/PRN max abs diff 0; DSU/MixStyle exact.')
if __name__=='__main__':main()
