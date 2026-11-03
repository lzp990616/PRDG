import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault('YOLO_AUTOINSTALL','False')
import argparse, time
import yaml
from ultralytics import YOLO
from prdg.training import trainer_for, disable_oom_retry
from prdg.io import DOMAINS, load, save, write_csv
from prdg.evaluation import evaluate
from scripts.normalize_dataset import prepare

def main():
    ap=argparse.ArgumentParser(description='Serial LODO reproduction. Best weights use source-only val.')
    ap.add_argument('--folds',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--domains',nargs='+',default=DOMAINS,choices=DOMAINS)
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,123,3407])
    ap.add_argument('--method',default='PRN+DSU')
    ap.add_argument('--model',default='yolo26n.pt',help='Official YOLO26n/YOLOv8n/YOLO11n checkpoint, or its local path')
    ap.add_argument('--protocol',choices=['main','screen'],default='main')
    ap.add_argument('--device',default='0');ap.add_argument('--workers',type=int,default=None)
    ap.add_argument('--smoke',action='store_true',help='One epoch only; never a paper result')
    a=ap.parse_args();root=Path(__file__).resolve().parents[1]
    methods=yaml.safe_load((root/'configs/methods.yaml').read_text());spec=methods[a.method]
    cfg=yaml.safe_load((root/'configs'/f'{a.protocol}.yaml').read_text())
    if a.workers is not None:cfg['workers']=a.workers
    if a.smoke:cfg.update(epochs=1,patience=1)
    if a.method=='Strong HSV':cfg.update({k:spec[k] for k in ('hsv_h','hsv_s','hsv_v')})
    a.out.mkdir(parents=True,exist_ok=False);completed=[]
    for domain in a.domains:
        fold=a.folds/domain
        if spec['preprocess']!='raw':
            # Fit the target once per fold on its training images, never on val/test.
            fold=prepare(fold,a.out/'inputs'/domain,spec['preprocess'])
        for seed in a.seeds:
            name=f'{domain}_seed{seed}';job=a.out/name;job.mkdir()
            settings={**cfg,'data':str((fold/'data.yaml').resolve()),'device':a.device,
                'seed':seed,'project':str(job.resolve()),'name':'fit','exist_ok':False,'resume':False}
            save(job/'config.json',settings)
            try:
                model=YOLO(a.model);model.add_callback('on_train_epoch_start',disable_oom_retry)
                model.train(trainer=trainer_for(a.method),**settings)
                weight=job/'fit/weights/best.pt'
                metrics=evaluate(weight,fold/'data.yaml',job/'heldout',a.device)
                row=dict(method=a.method,source=domain,seed=seed,epochs=cfg['epochs'],smoke=a.smoke,**metrics)
                completed.append(row);save(job/'row.json',row)
                # Append immediately; an exception never removes a previous result.
                import csv
                path=a.out/'completed_evaluations.csv'
                with path.open('a',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(row))
                    if f.tell()==0:w.writeheader()
                    w.writerow(row)
            except Exception:
                import traceback
                (job/'ERROR.md').write_text(traceback.format_exc(),encoding='utf-8');raise
    save(a.out/'COMPLETE.json',dict(completed=len(completed),smoke=a.smoke))
if __name__=='__main__':main()
