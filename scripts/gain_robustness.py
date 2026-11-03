import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, shutil
import cv2, numpy as np, pandas as pd
from prdg.io import rows, load, save, write_csv, read_image, image_path, label_path, write_yaml
from prdg.normalization import normalize
from prdg.baselines.references import transform
from prdg.evaluation import evaluate
GAINS=[(1,1,1),(.8,.8,.8),(.7,.85,1),(1,.85,.7),(.85,1,.85),(.9,.9,1.1),(1.1,.95,.85)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights-csv',type=Path,required=True,help='source,method,seed,weights,fold,target_json; paths relative to CSV')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='0');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    all_rows=[];base=a.weights_csv.resolve().parent
    for entry in rows(a.weights_csv):
        fold=base/entry['fold'];weights=base/entry['weights'];target=load(base/entry['target_json'])['target_rgb'] if entry.get('target_json') else None
        selected=[r for r in rows(fold/'manifest.csv') if r['loso_split']=='test']
        for gi,gain in enumerate(GAINS):
            job=a.out/f"{entry['source']}_{entry['method']}_{entry['seed']}_{gi}";data=job/'data';data.mkdir(parents=True)
            for r in selected:
                im=read_image(image_path(fold,r));rgb=im[:,:,::-1].astype(np.float32)
                changed=np.rint(np.clip(rgb*np.asarray(gain,np.float32),0,255)).astype(np.uint8)[:,:,::-1].copy()
                if target is not None:
                    changed=transform(changed,entry['method'],target)[0] if entry['method'] in ('B0','B1','B5','B6') else normalize(changed,target)
                dest=image_path(data,r);dest.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(dest),changed)
                lab=label_path(data,r);lab.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(label_path(fold,r),lab)
            # val aliases test only for this explicit final-evaluation dataset.
            write_yaml(data)
            import yaml
            y=yaml.safe_load((data/'data.yaml').read_text());y.update(train='images/test',val='images/test');(data/'data.yaml').write_text(yaml.safe_dump(y))
            metric=evaluate(weights,data/'data.yaml',job/'evaluation',a.device)
            row={k:entry[k] for k in ('source','method','seed')}|dict(gain_id=gi,R=gain[0],G=gain[1],B=gain[2],**metric)
            all_rows.append(row);write_csv(job/'result.csv',[row])
    write_csv(a.out/'gain_results.csv',all_rows)
    df=pd.DataFrame(all_rows);spread=df.groupby(['method','source','seed']).map50_95.agg(lambda x:x.max()-x.min()).reset_index(name='range')
    spread.to_csv(a.out/'ranges.csv',index=False)
    spread.groupby(['method','source']).range.agg(['mean','std']).to_csv(a.out/'ranges_mean_sd.csv')
if __name__=='__main__':main()
