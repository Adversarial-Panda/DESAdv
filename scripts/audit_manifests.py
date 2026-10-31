"""Check pairing, image SHA-256 overlap and joint-benign-correct evidence CSV.

Eligibility evidence requires one row per sample/target with clean_correct=1;
never use target feedback in selector or generation scripts.
"""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--file-list',required=True)
    p.add_argument('--eligibility',help='CSV with sample_id,target,clean_correct for 9/7 targets')
    p.add_argument('--expected-count',type=int,default=1000)
    p.add_argument('--expected-targets',type=int,default=9)
    a=p.parse_args()
    paths=[x.strip() for x in Path(a.file_list).read_text().splitlines() if x.strip()]
    if len(paths)!=a.expected_count or len(paths)!=len(set(paths)):
        raise ValueError('Wrong image count or duplicate paths')
    missing=[x for x in paths if not Path(x).is_file()]
    if missing:raise FileNotFoundError(missing[:3])
    digests=[hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in paths]
    if len(set(digests))!=len(digests):raise ValueError('Duplicate content inside selected query list')
    result={'images':len(paths),'file_list_sha256':hashlib.sha256(Path(a.file_list).read_bytes()).hexdigest(),
            'joint_target_correct_verified':False}
    if a.eligibility:
        with Path(a.eligibility).open(newline='') as f:rows=list(csv.DictReader(f))
        grouped=defaultdict(dict)
        for r in rows:
            s=r['sample_id'];t=r['target'];y=r['clean_correct']
            if t in grouped[s] or y not in ('0','1'):raise ValueError('Duplicate/malformed eligibility row')
            grouped[s][t]=int(y)
        if len(grouped)!=len(paths) or any(len(targets)!=a.expected_targets or not all(targets.values()) for targets in grouped.values()):
            raise ValueError('Not all selected images are correct for every target')
        result['joint_target_correct_verified']=True
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
