"""Small filesystem helpers. Outputs must never alias input data."""
import csv, hashlib, json
from pathlib import Path
import cv2, yaml
NAMES = ['WBC', 'RBC', 'Platelets']
DOMAINS = ['BCCD', 'PBC', 'Raabin-WBC', 'BCDD']
def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def write_csv(path, records):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not records: raise ValueError('No records')
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_image(path):
    image=cv2.imread(str(path))
    if image is None:raise ValueError('Unable to decode image: '+str(path))
    return image
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_yaml(folder):
    folder=Path(folder).resolve()
    obj=dict(path=folder.as_posix(),train='images/train',val='images/val',test='images/test',names=NAMES,nc=3)
    (folder/'data.yaml').write_text(yaml.safe_dump(obj,sort_keys=False),encoding='utf-8')
def image_path(folder,row):return Path(folder)/'images'/row['loso_split']/row['new_name']
def label_path(folder,row):return Path(folder)/'labels'/row['loso_split']/(Path(row['new_name']).stem+'.txt')
