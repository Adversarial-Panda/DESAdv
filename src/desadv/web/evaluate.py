"""Paired conditional Top-k ASR from cached API/VLM JSONL responses.

Required input rows: image_id, method, split (clean/adv), label, predictions.
The denominator at each k is the number of BENIGN-CORRECT clean responses;
missing adv responses are an ERROR, not an unsuccessful attack or an exclusion.
"""
import argparse, json, csv
from collections import defaultdict
from pathlib import Path
from .matching import topk_correct

def load_jsonl(path):
    with open(path,encoding='utf8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def evaluate(rows, ks=(1,5,10), mode='lexical_overlap', aliases=None):
    groups=defaultdict(dict)
    for row in rows:
        needed={'image_id','method','split','label','predictions','provider'}
        if not needed.issubset(row): raise ValueError(f'Missing fields: {sorted(needed-set(row))}')
        if row['split'] not in ('clean','adv'): raise ValueError('split must be clean or adv')
        if not isinstance(row['predictions'],list): raise ValueError('predictions must be ordered label list')
        key=(row['provider'],row['method'],str(row['image_id']))
        if row['split'] in groups[key]: raise ValueError(f'Duplicate prediction for {key} {row["split"]}')
        groups[key][row['split']]=row
    pairs=defaultdict(list)
    for (provider,method,imgid),pair in groups.items():
        if set(pair)!= {'clean','adv'}: raise ValueError(f'Unpaired prediction: {provider}/{method}/{imgid}: {set(pair)}')
        if pair['clean']['label']!=pair['adv']['label']: raise ValueError(f'Label mismatch: {imgid}')
        pairs[(provider,method)].append(pair)
    output=[]
    for (provider,method),vals in sorted(pairs.items()):
        for k in ks:
            n=sum(topk_correct(p['clean']['label'],p['clean']['predictions'],k,mode=mode,aliases=aliases) for p in vals)
            hits=sum(topk_correct(p['clean']['label'],p['clean']['predictions'],k,mode=mode,aliases=aliases)
                     and not topk_correct(p['adv']['label'],p['adv']['predictions'],k,mode=mode,aliases=aliases) for p in vals)
            output.append(dict(provider=provider,method=method,k=k,N=len(vals),benign_correct=n,
                               benign_agreement_pct=round(100*n/len(vals),4),
                               attack_success_n=hits,
                               conditional_asr_pct=round(100*hits/n,4) if n else None,
                               match_rule=mode))
    return output

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--responses',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--ks',default='1,5,10')
    a.add_argument('--match-rule',choices=['lexical_overlap','exact'],default='lexical_overlap')
    a.add_argument('--aliases',help='Optional frozen JSON object mapping GT label to list of aliases')
    args=a.parse_args()
    aliases=json.loads(Path(args.aliases).read_text()) if args.aliases else None
    summary=evaluate(load_jsonl(args.responses),[int(t) for t in args.ks.split(',')],args.match_rule,aliases)
    if not summary: raise ValueError('No paired predictions')
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
