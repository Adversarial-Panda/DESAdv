"""Offline zero-shot CLIP evaluation on OWN labeled 3x3 image-tile datasets.

Input JSONL rows have session_id, target_label, tiles (9 paths), correct_indices.
This is a controlled study, not a solver for public CAPTCHA systems.
"""
import argparse, json
from pathlib import Path

def solve(rows,model_id='openai/clip-vit-base-patch32',threshold=0.25):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=CLIPModel.from_pretrained(model_id).to(dev).eval()
    p=CLIPProcessor.from_pretrained(model_id)
    classes=['goldfish','tiger','zebra','school bus','banana','ant',
             'pizza','bee','traffic light','elephant']
    out=[]
    for item in rows:
        paths=item['tiles']
        if len(paths)!=9: raise ValueError('Expected nine local tile paths')
        imgs=[]
        for path in paths:
            with Image.open(path) as im:imgs.append(im.convert('RGB').copy())
        if item['target_label'] not in classes:
            raise ValueError('Unknown target: expected one of the 10 paper class names')
        prompts=[f'a photo of a {name}' for name in classes]
        batch=p(text=prompts,images=imgs,return_tensors='pt',padding=True).to(dev)
        with torch.inference_mode():
            z=m(**batch).logits_per_image.softmax(dim=1).detach().cpu()
        target=classes.index(item['target_label'])
        # Threshold is UNCALIBRATED. Tune with disjoint validation sessions.
        predicted=[i for i,s in enumerate(z) if int(s.argmax())==target and float(s[target])>=threshold]
        correct=sorted(int(i) for i in item['correct_indices'])
        out.append({'solver':'CLIP ViT-B/32','layout':'grid',
                    'condition':item.get('condition','clean'),
                    'session_id':item['session_id'], 'solved':predicted==correct,
                    'predicted_indices':predicted,'threshold':threshold})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,default=.25)
    a=p.parse_args();rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    out=solve(rows,threshold=a.threshold)
    Path(a.output).parent.mkdir(exist_ok=True,parents=True)
    Path(a.output).write_text(''.join(json.dumps(x)+'\n' for x in out))
if __name__=='__main__':main()
