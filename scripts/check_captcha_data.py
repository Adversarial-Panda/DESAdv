#!/usr/bin/env python3
"""Check custom CAPTCHA image manifests for same-filename class pairing.

Does not infer the attack epsilon or whether images are genuinely adversarial.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FOLDERS = {'benign':'captcha_images', 'adv':'captcha_images_adv', 'adv2':'CAPTCHA_ADV_IMAGES_2'}

def audit(root, require_all=False):
    root = Path(root) / 'dataset'
    manifest = root / 'captcha_images_labels.csv'
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        data = list(csv.DictReader(f))
    if not data or not {'image_name','label'} <= set(data[0]):
        raise ValueError('Manifest requires image_name,label and at least one row')
    names = [r['image_name'] for r in data]
    if len(set(names)) != len(names) or any(not p or Path(p).name != p for p in names):
        raise ValueError('Duplicate or invalid image names in label manifest')
    labels=Counter(row['label'] for row in data)
    results={'labeled_images':len(names),'classes':dict(sorted(labels.items())),'conditions':{}}
    for condition, folder in FOLDERS.items():
        folder_path = root/folder
        if not folder_path.exists():
            if require_all:raise ValueError(f'Missing images: {folder_path}')
            results['conditions'][condition]={'status':'not_provided'}
            continue
        existing={p.name for p in folder_path.iterdir() if p.is_file()}
        missing=sorted(set(names)-existing)
        extra=sorted(existing-set(names))
        results['conditions'][condition]={'status':'complete' if not (missing or extra) else 'mismatch',
                                           'count':len(existing),'missing':missing[:10],
                                           'extra':extra[:10], 'missing_count':len(missing),'extra_count':len(extra)}
        if missing or extra:raise ValueError(f'Image filename mismatch for {condition}: {len(missing)} missing, {len(extra)} extra')
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='experiments/captcha')
    p.add_argument('--require-all',action='store_true')
    args=p.parse_args()
    print(json.dumps(audit(args.root, args.require_all),indent=2))

if __name__=='__main__': main()
