"""Prepare exact Table 8 fixed-set jobs without invalidly mixing settings.json.

Dry run by default. Explicit --execute launches jobs sequentially. Each set
gets an isolated out_dir but shares copies of prepared/scores caches.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from desadv.fixed_sets import FIXED_SETS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--test-list',required=True)
    p.add_argument('--val-dir',required=True)
    p.add_argument('--methods',default='ens,svre,adaea,smer')
    p.add_argument('--optimization',choices=['mifgsm','pgd'],default='mifgsm')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    cache=Path(a.prepared_dir)
    for name in ['prepared.pt','scores_paper_v1.pt']:
        if not (cache/name).exists():raise FileNotFoundError(cache/name)
    for number,indices in FIXED_SETS.items():
        out=Path(a.output_root)/f'set_{number:02d}'
        cmd=[sys.executable,str(ROOT/'experiments/paper_ensemble.py'),'attack',
             '--out-dir',str(out),'--test-list',a.test_list,'--val-dir',a.val_dir,
             '--methods',a.methods,'--optimization',a.optimization,
             '--variants','fixed','--fixed-indices',','.join(map(str,indices)),
             '--limit',str(a.limit)]
        print(' '.join(cmd),flush=True)
        if a.execute:
            out.mkdir(exist_ok=True,parents=True)
            for name in ['prepared.pt','scores_paper_v1.pt']:
                if not (out/name).exists():shutil.copy2(cache/name,out/name)
            subprocess.run(cmd,check=True)
if __name__=='__main__':main()
