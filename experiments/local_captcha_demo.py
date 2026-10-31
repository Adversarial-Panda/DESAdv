"""Self-hosted Gradio visual-challenge demo for controlled evaluation.

Use only a CSV of images you own, with columns `path,label`. Supports nine-tile
(grid) and tiled collage presentation; human selection uses checkbox indices.
The demo and browser solver have separate input contracts.
"""
import argparse, base64, csv, io, json, random, uuid
from pathlib import Path
from PIL import Image,ImageDraw

def load_items(csv_path):
    with open(csv_path,newline='') as f: items=list(csv.DictReader(f))
    items=[row for row in items if row.get('path') and row.get('label') and Path(row['path']).is_file()]
    if len(items)<9 or len(set(r['label'] for r in items))<2:
        raise ValueError('Need >=9 labeled image paths from >=2 target categories')
    return items

def challenge(items, layout, rng):
    selected=rng.sample(items,9)
    target=rng.choice(selected)['label']
    correct=[i for i,r in enumerate(selected) if r['label']==target]
    thumbs=[]
    for r in selected:
        with Image.open(r['path']) as im:
            thumbs.append(im.convert('RGB').resize((160,160)).copy())
    canvas=Image.new('RGB',(480,480),'white')
    for i,im in enumerate(thumbs):
        canvas.paste(im,((i%3)*160,(i//3)*160))
    draw=ImageDraw.Draw(canvas)
    for i in range(9):
        draw.text(((i%3)*160+4,(i//3)*160+5),str(i),fill='white',stroke_width=2,stroke_fill='black')
    return canvas,target,correct

def app(csv_path,log_file,seed=42):
    import gradio as gr
    images=load_items(csv_path);rng=random.Random(seed)
    dest=Path(log_file);dest.parent.mkdir(exist_ok=True,parents=True)
    def new(layout):
        picture,target,answers=challenge(images,layout,rng)
        state={'session_id':str(uuid.uuid4()),'layout':layout,'target':target,'correct_indices':answers}
        return picture, f'Select all tiles containing **{target}** (0–8)',[],state,''
    def verify(selected,state):
        if not state: return 'Create a challenge first.'
        try: chosen=sorted(int(x) for x in selected)
        except Exception:chosen=[]
        solved=chosen==state['correct_indices']
        row=dict(session_id=state['session_id'],solver='human_demo',layout=state['layout'],
                 condition='clean',solved=bool(solved),selected_indices=chosen)
        with dest.open('a') as f:f.write(json.dumps(row)+'\n')
        return 'Correct' if solved else 'Incorrect'
    with gr.Blocks(title='Local DES-Adv research CAPTCHA demo') as demo:
        gr.Markdown('# Local visual challenge demonstration\nOnly for research on challenges you own.')
        layout=gr.Radio(['grid','collage'],value='grid',label='Layout')
        pic=gr.Image(label='3 × 3 challenge (each tile indexed 0–8)',interactive=False)
        prompt=gr.Markdown()
        selected=gr.CheckboxGroup([str(i) for i in range(9)],label='Selected tile indices')
        st=gr.State();result=gr.Markdown()
        fresh=gr.Button('New local challenge');check=gr.Button('Verify selection')
        fresh.click(new,inputs=layout,outputs=[pic,prompt,selected,st,result])
        check.click(verify,inputs=[selected,st],outputs=result)
    demo.launch(server_name='127.0.0.1',share=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',required=True,help='CSV columns path,label')
    p.add_argument('--log-file',default='outputs/local_captcha_demo.jsonl')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();app(a.images,a.log_file,a.seed)
if __name__=='__main__':main()
