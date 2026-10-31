"""Evaluation-only adapters. Network calls are never used for attack construction.

Model identifiers are explicit configurable strings, not frozen assumptions about
what remote services currently expose. Provider SDK availability varies by date.
"""
import base64, io, json, os
from pathlib import Path
from PIL import Image

PROMPT = ("Examine the image and identify the main object shown. "
          "Return your five most likely object categories in ranked order "
          'as JSON: {"top5": ["category1", "category2", "category3", '
          '"category4", "category5"]}. Do not provide an explanation.')

def _png_bytes(path):
    """Decode and re-encode to valid PNG; hash and provenance are logged upstream."""
    with Image.open(path) as image:
        buff=io.BytesIO();image.convert('RGB').save(buff,format='PNG')
    return buff.getvalue()

def parse_top5(text):
    s=text.strip()
    if s.startswith('```'):
        s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
    parsed=json.loads(s)
    if not isinstance(parsed,dict) or list(parsed.keys())!=['top5']:
        raise ValueError('VLM output must be JSON object with only top5')
    labels=parsed['top5']
    if not isinstance(labels,list) or len(labels)!=5 or any(not isinstance(v,str) or not v.strip() for v in labels):
        raise ValueError('Top5 must contain exactly five nonempty string categories')
    return [v.strip() for v in labels]

def query(provider, image_path, model=None, region='us-east-1'):
    """Return ranked textual label list; errors propagate and do NOT count as ASR."""
    data=_png_bytes(image_path)
    if provider=='aws':
        import boto3
        result=boto3.client('rekognition',region_name=region).detect_labels(
            Image={'Bytes':data},MaxLabels=10,MinConfidence=0)
        return [x['Name'] for x in result.get('Labels',[])][:10]
    if provider=='google_vision':
        from google.cloud import vision
        client=vision.ImageAnnotatorClient()
        resp=client.label_detection(image=vision.Image(content=data),max_results=10)
        if resp.error.message: raise RuntimeError(resp.error.message)
        return [x.description for x in resp.label_annotations][:10]
    if provider=='imagga':
        import requests
        key,secret=os.environ.get('IMAGGA_API_KEY'),os.environ.get('IMAGGA_API_SECRET')
        if not key or not secret:raise EnvironmentError('IMAGGA_API_KEY and IMAGGA_API_SECRET required')
        r=requests.post('https://api.imagga.com/v2/tags',files={'image':('image.png',data,'image/png')},
                        params={'language':'en','limit':10},auth=(key,secret),timeout=60)
        r.raise_for_status()
        return [entry['tag']['en'] for entry in r.json()['result']['tags']][:10]
    if provider=='hf_convnext':
        from huggingface_hub import InferenceClient
        client=InferenceClient(token=os.getenv('HF_TOKEN'))
        result=client.image_classification(data,model=model or 'facebook/convnext-base-224',top_k=10)
        return [x.label for x in sorted(result,key=lambda x:-x.score)][:10]
    if provider=='openai':
        from openai import OpenAI
        if not model:raise ValueError('--model required for OpenAI; paper used gpt-5.6')
        encoded=base64.b64encode(data).decode('ascii')
        response=OpenAI().responses.create(model=model,input=[{'role':'user','content':[
            {'type':'input_text','text':PROMPT},
            {'type':'input_image','image_url':'data:image/png;base64,'+encoded}]}])
        return parse_top5(response.output_text)
    if provider=='anthropic':
        import anthropic
        if not model:raise ValueError('--model required for Anthropic; paper used claude-sonnet-4-6')
        result=anthropic.Anthropic().messages.create(model=model,max_tokens=256,messages=[
            {'role':'user','content':[
              {'type':'image','source':{'type':'base64','media_type':'image/png',
                                      'data':base64.b64encode(data).decode('ascii')}},
              {'type':'text','text':PROMPT}]}])
        content=''.join(c.text for c in result.content if getattr(c,'type',None)=='text')
        return parse_top5(content)
    if provider=='gemini':
        from google import genai
        from google.genai import types
        if not model:raise ValueError('--model required for Gemini; paper used gemini-3.6-flash')
        result=genai.Client().models.generate_content(
            model=model,contents=[PROMPT,Image.open(io.BytesIO(data))],
            config=types.GenerateContentConfig(temperature=0,response_mime_type='application/json'))
        return parse_top5(result.text)
    raise ValueError('Unknown provider: '+provider)
