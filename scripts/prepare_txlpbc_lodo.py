import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, shutil, json, hashlib, random
import pandas as pd
from prdg.io import write_csv, write_yaml
SOURCES=['BCCD','BCDD','PBC','Raabin-WBC']
def stable_seed(base_seed: int, source: str) -> int:
    h = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return base_seed + (int(h[:8], 16) % 100000)

def scan_dataset(root: Path):
    df = pd.read_excel(root / "metadata_file.xlsx")
    req = {"Original_Image_Name", "Source_Dataset", "New_Image_Name", "Source_URL"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing metadata columns: {sorted(miss)}")

    image_map = {}
    for split in ["train", "val", "test"]:
        idir = root / "images" / split
        ldir = root / "labels" / split
        if not idir.exists() or not ldir.exists():
            raise FileNotFoundError(f"Missing images/labels directory for split={split}")
        for p in idir.iterdir():
            if p.is_file():
                if p.name in image_map:
                    raise RuntimeError(f"Duplicate image filename across official splits: {p.name}")
                image_map[p.name] = (split, p.resolve())

    rows = []
    missing_images, missing_labels = [], []

    for _, r in df.iterrows():
        name = str(r["New_Image_Name"])
        source = str(r["Source_Dataset"])
        if source not in SOURCES:
            raise ValueError(f"Unexpected source: {source}")
        if name not in image_map:
            missing_images.append(name)
            continue
        official_split, img_path = image_map[name]
        label_path = (root / "labels" / official_split / f"{Path(name).stem}.txt").resolve()
        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        rows.append({
            "original_name": str(r["Original_Image_Name"]),
            "source": source,
            "new_name": name,
            "official_split": official_split,
            "image_path": str(img_path),
            "label_path": str(label_path),
        })

    if missing_images or missing_labels:
        raise RuntimeError(
            f"Integrity failure: missing_images={len(missing_images)}, "
            f"missing_labels={len(missing_labels)}"
        )
    if len(rows) != len(df):
        raise RuntimeError(f"Mapped {len(rows)} / {len(df)} metadata rows.")

    mapped = pd.DataFrame(rows)
    if mapped["new_name"].duplicated().any():
        raise RuntimeError("Duplicate New_Image_Name detected after mapping.")
    return df, mapped

def make_source_internal_splits(mapped: pd.DataFrame, val_frac: float, split_seed: int):
    """
    Precompute ONE fixed internal train/val partition per source.
    The same source partition is reused in every LOSO fold where that source is training data.
    """
    assignments = {}
    audit = []
    for source in SOURCES:
        names = sorted(mapped.loc[mapped.source == source, "new_name"].tolist())
        rng = random.Random(stable_seed(split_seed, source))
        rng.shuffle(names)

        n_val = max(1, int(round(len(names) * val_frac)))
        val = set(names[:n_val])
        train = set(names[n_val:])

        if train & val:
            raise RuntimeError(f"Internal overlap for {source}")

        assignments[source] = {"train": train, "val": val}
        audit.append({
            "source": source,
            "n_total": len(names),
            "n_internal_train": len(train),
            "n_internal_val": len(val),
            "val_fraction_actual": len(val) / len(names),
            "seed": stable_seed(split_seed, source),
        })
    return assignments, pd.DataFrame(audit)

def main():
    ap=argparse.ArgumentParser(description='Exact paper LODO: fixed per-source seed42, 15% internal validation.')
    ap.add_argument('--root',type=Path,required=True,help='Official TXL-PBC directory containing metadata_file.xlsx')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();out=args.out.resolve()
    if out==root or root in out.parents:raise ValueError('Output must be outside the input dataset')
    out.mkdir(parents=True,exist_ok=False)
    _,mapped=scan_dataset(root)
    assignments,audit=make_source_internal_splits(mapped,.15,42)
    expected={'BCCD':(935,165,160),'PBC':(646,114,500),'Raabin-WBC':(646,114,500),'BCDD':(986,174,100)}
    for held in SOURCES:
        folder=out/held;records=[]
        for row in mapped.to_dict('records'):
            split='test' if row['source']==held else ('val' if row['new_name'] in assignments[row['source']]['val'] else 'train')
            for kind,source,suffix in [('images',row['image_path'],Path(row['new_name']).suffix),('labels',row['label_path'],'.txt')]:
                dest=folder/kind/split/(Path(row['new_name']).stem+suffix)
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            records.append({k:row[k] for k in ('source','new_name','official_split')}|{'loso_split':split})
        counts=tuple(sum(r['loso_split']==s for r in records) for s in ('train','val','test'))
        if counts!=expected[held]:raise ValueError((held,counts,expected[held]))
        write_csv(folder/'manifest.csv',records);write_yaml(folder)
    audit.to_csv(out/'split_audit.csv',index=False)
    print('Prepared four folds. Copies only; no hard links or source cache writes.')
if __name__=='__main__':main()
