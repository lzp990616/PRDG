import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
from prdg.io import rows,save
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original-splits',type=Path,required=True);ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();report=[]
    for domain,old in [('BCCD','bccd'),('PBC','pbc'),('Raabin-WBC','raabin_wbc'),('BCDD','bcdd')]:
        expected=rows(a.original_splits/f'loso_{old}/split_manifest.csv');actual=rows(a.prepared/domain/'manifest.csv')
        key=lambda rr:[(r['new_name'],r['source'],r['loso_split']) for r in rr]
        assert key(expected)==key(actual)
        report.append(dict(domain=domain,exact_order_and_membership=True,**{s:sum(r['loso_split']==s for r in actual) for s in ('train','val','test')}))
    save(a.out,dict(passed=True,folds=report));print('PASS: all four original split memberships and order match.')
if __name__=='__main__':main()
