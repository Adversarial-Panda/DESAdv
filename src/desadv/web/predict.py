"""Collect CLEAN/ADV paired responses; attack must be generated OFFLINE first.

CSV input: image_id,method,split,label,path. There MUST be a clean and adv
record for each (method,image_id). No remote queries occur without --allow-live.
"""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
from .providers import query

PROVIDERS=('aws','google_vision','imagga','hf_convnext','openai','anthropic','gemini')

def read_manifest(path):
    with Path(path).open(newline='',encoding='utf8') as file:
        rows=list(csv.DictReader(file))
    if not rows:raise ValueError('Empty manifest')
    keys={}
    for row in rows:
        for name in ('image_id','method','split','label','path'):
            if not row.get(name): raise ValueError(f'Missing manifest field: {name}')
        if row['split'] not in ('clean','adv'):raise ValueError('split must be clean or adv')
        key=(row['method'],row['image_id'])
        if row['split'] in keys.setdefault(key,set()):raise ValueError(f'Duplicate split: {key}')
        keys[key].add(row['split'])
    if any(v!={'clean','adv'} for v in keys.values()):raise ValueError('Every image/method needs both clean and adv')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--provider',required=True,choices=PROVIDERS)
    p.add_argument('--model',help='Exact deployed model ID (VLMs require it)')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=10,help='Number of paired IDs; default 10 for billing safety')
    p.add_argument('--region',default='us-east-1')
    p.add_argument('--allow-live',action='store_true',help='Explicitly authorize paid target queries for evaluation only')
    p.add_argument('--sleep-seconds',type=float,default=0)
    args=p.parse_args()
    if args.limit<1 or args.sleep_seconds<0:raise ValueError('Invalid limit/sleep')
    rows=read_manifest(args.manifest)
    keys=list(dict.fromkeys((r['method'],r['image_id']) for r in rows))[:args.limit]
    chosen=set(keys)
    rows=[r for r in rows if (r['method'],r['image_id']) in chosen]
    missing=[r['path'] for r in rows if not Path(r['path']).is_file()]
    if missing:raise FileNotFoundError(f'Missing images: {missing[:5]}')
    print(f'Provider={args.provider}, model={args.model}, paired={len(keys)}, calls={len(rows)}')
    if not args.allow_live:
        print('DRY RUN ONLY; add --allow-live to authorize network calls and potential charges.')
        return
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    seen=set()
    clean_cache={}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                old=json.loads(line)
                seen.add((old['provider'],old['method'],old['image_id'],old['split']))
                if old.get('model')!=args.model:raise ValueError('Existing output uses another model; choose new file')
                if old['split']=='clean':
                    clean_cache[(old['image_id'],old['label'],old['source_sha256'])]=old['predictions']
    with out.open('a',encoding='utf8') as f:
        for r in rows:
            key=(args.provider,r['method'],r['image_id'],r['split'])
            if key in seen:continue
            started=time.time()
            source_sha=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
            clean_key=(r['image_id'],r['label'],source_sha)
            reused_clean=r['split']=='clean' and clean_key in clean_cache
            preds=clean_cache[clean_key] if reused_clean else query(args.provider,r['path'],model=args.model,region=args.region)
            if r['split']=='clean':clean_cache[clean_key]=preds
            row={k:r[k] for k in ('image_id','method','split','label')}
            row.update(provider=args.provider,model=args.model,region=args.region,
                       predictions=preds,source_sha256=source_sha,reused_clean_response=reused_clean,
                       elapsed_s=round(time.time()-started,4),timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            time.sleep(args.sleep_seconds)
    print('Saved complete responses to',out)
if __name__=='__main__':main()
