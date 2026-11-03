import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from prdg.evaluation import evaluate
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='0');ap.add_argument('--split',default='test',choices=['test','val'])
    a=ap.parse_args();print(evaluate(a.weights,a.data,a.out,a.device,a.split))
if __name__=='__main__':main()
