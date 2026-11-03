import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from prdg.io import write_csv
METRICS=['map50_95','map50','map75','WBC_AP','RBC_AP','Platelets_AP']
def paired(x):
    x=np.asarray(x,float);n=len(x)
    ci=np.quantile(np.random.default_rng(0).choice(x,(100000,n),replace=True).mean(1),[.025,.975])
    return dict(n=n,wins=int((x>0).sum()),delta=float(x.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
        wilcoxon_p=1. if np.all(x==0) else float(wilcoxon(x,alternative='two-sided',zero_method='wilcox',method='auto').pvalue))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',nargs='+',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',default='Baseline');a=ap.parse_args()
    df=pd.concat([pd.read_csv(p) for p in a.csv],ignore_index=True)
    if 'smoke' in df and df.smoke.fillna(False).astype(str).str.lower().isin(['true','1']).any():raise ValueError('Smoke results cannot enter paper statistics')
    if 'epochs' in df and df.epochs.nunique()>1:raise ValueError('Do not mix training protocols')
    if df.duplicated(['method','source','seed']).any():raise ValueError('Duplicate setting; resolve provenance explicitly')
    a.out.mkdir(parents=True,exist_ok=False);metrics=[k for k in METRICS if k in df];summary=[];tests=[]
    for method,group in df.groupby('method'):
        blocks=list(group.groupby('source'))+[('Four-domain average',group.groupby('seed')[metrics].mean().reset_index())]
        for source,g in blocks:
            row=dict(method=method,source=source,n=len(g))
            for metric in metrics:row.update({metric+'_mean':g[metric].mean(),metric+'_sd':g[metric].std(ddof=1)})
            summary.append(row)
        if method==a.reference:continue
        ref=df[df.method==a.reference];matched=group.merge(ref,on=['source','seed'],suffixes=('_a','_b'),validate='one_to_one')
        for metric in metrics:
            valid=matched[[metric+'_a',metric+'_b']].dropna()
            if len(valid):tests.append(dict(method=method,reference=a.reference,metric=metric,**paired(valid[metric+'_a']-valid[metric+'_b'])))
    write_csv(a.out/'mean_sd.csv',summary)
    if tests:write_csv(a.out/'paired.csv',tests)
if __name__=='__main__':main()
