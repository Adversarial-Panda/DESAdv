"""Calculate custom local CAPTCHA CSR from session-level JSONL.

No interaction with public/third-party CAPTCHA deployments. Required rows:
method,layout,condition,session_id,solved (bool). Protocol expects 500 sessions
per condition, but this evaluator always reports the actual denominator.
"""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path

def csr(rows):
    g=defaultdict(list)
    ids=set()
    for r in rows:
        for k in ('solver','layout','condition','session_id','solved'):
            if k not in r: raise ValueError(f'Missing {k}')
        if not isinstance(r['solved'],bool): raise ValueError('solved must be JSON boolean')
        ident=(r['solver'],r['layout'],r['condition'],str(r['session_id']))
        if ident in ids: raise ValueError('duplicate session')
        ids.add(ident);g[ident[:3]].append(r['solved'])
    return [dict(solver=s,layout=l,condition=c,sessions=len(x),successes=sum(x),
                 csr_pct=round(100*sum(x)/len(x),4)) for (s,l,c),x in sorted(g.items())]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sessions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.sessions).read_text().splitlines() if line.strip()]
    result=csr(rows)
    if not result: raise ValueError('No sessions')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0].keys());w.writeheader();w.writerows(result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
