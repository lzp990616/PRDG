"""Official Ultralytics FP32 validation; no custom AP implementation."""
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from .io import NAMES, save, sha256
from . import dsu
from .baselines import mixstyle

class FP32Validator(DetectionValidator):
    def preprocess(self,batch):
        batch=super().preprocess(batch)
        assert batch['img'].dtype==torch.float32
        return batch

def evaluate(weights,data,output,device='0',split='test'):
    # Historical pickles use these module names. No legacy project import.
    sys.modules.setdefault('dsu_shallow',dsu)
    sys.modules.setdefault('mixstyle_shallow',mixstyle)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=YOLO(str(weights)).val(data=str(data),split=split,imgsz=640,batch=2,
        half=False,quantize=None,conf=.001,iou=.7,max_det=300,device=device,
        workers=0,plots=False,verbose=False,validator=FP32Validator,
        project=str(output),name='validation',exist_ok=False,
        save_json=False,save_txt=False,augment=False)
    box=result.box
    metrics=dict(map50_95=float(box.map),map50=float(box.map50),map75=float(box.map75),
        precision=float(box.mp),recall=float(box.mr))
    # Missing classes are NA, never fill with the overall mAP.
    present={int(c):float(box.ap[i]) for i,c in enumerate(box.ap_class_index)}
    metrics.update({c+'_AP':present.get(i) for i,c in enumerate(NAMES)})
    save(output/'metrics.json',dict(metrics=metrics,weight_sha256=sha256(weights),
        protocol=dict(imgsz=640,batch=2,half=False,conf=.001,iou=.7,max_det=300,split=split)))
    return metrics
