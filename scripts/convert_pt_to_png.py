"""Convert numbered [3,H,W] float .pt adversarial tensors to lossless PNGs.

Enforces 1:1 clean/adv list pairing by index. Quantization affects perturbation
bounds; this script does NOT certify post-encoding epsilon, which must be checked.
"""
import argparse,csv,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pt-dir',required=True)
    p.add_argument('--clean-list',required=True,help='Text file of clean image paths in the EXACT attack sample order')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--trust-pt',action='store_true',help='User-owned torch tensors only')
    a=p.parse_args()
    if not a.trust_pt:raise ValueError('--trust-pt required: only open known, trusted torch files')
    clean=[x.strip() for x in Path(a.clean_list).read_text().splitlines() if x.strip()]
    if not clean:raise ValueError('Empty clean list')
    files=sorted(Path(a.pt_dir).glob('*.pt'))
    if {x.name for x in files}!={f'{i}.pt' for i in range(len(clean))}:
        raise ValueError('Expected EXACT numbered 0.pt..N-1.pt pairing and no extra .pt files')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for i,cp in enumerate(clean):
        if not Path(cp).exists():raise FileNotFoundError(cp)
        # weights_only blocks arbitrary object deserialization in current torch.
        raw=torch.load(Path(a.pt_dir)/f'{i}.pt',map_location='cpu',weights_only=True)
        if not isinstance(raw,torch.Tensor) or raw.ndim!=3 or raw.shape[0]!=3:
            raise ValueError(f'{i}.pt is not a CHW image tensor')
        if not torch.isfinite(raw).all() or raw.min()<0 or raw.max()>1:
            raise ValueError(f'{i}.pt outside [0,1]')
        arr=(raw.permute(1,2,0).numpy()*255).round().astype(np.uint8)
        adv=out/f'adv_{i:04d}.png'
        Image.fromarray(arr,'RGB').save(adv)
        # Ground-truth ImageNet label can be from WNID parent, but lexical
        # matching requires a descriptive class label: explicit fill is mandatory.
        for split,path in [('clean',cp),('adv',str(adv))]:
            manifest.append(dict(image_id=str(i),method=a.method,split=split,label='',path=str(Path(path).resolve())))
    with (out/'web_manifest_NEEDS_LABELS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image_id','method','split','label','path'])
        w.writeheader();w.writerows(manifest)
    print('Wrote',len(clean),'lossless PNG adversarial files and manifest; fill labels and verify final epsilon before use.')
if __name__=='__main__':main()
