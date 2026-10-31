"""Lexical label agreement for Web API concept predictions.

Matching is applied identically to clean/adversarial responses. Use frozen aliases
for every method; DO NOT tune matching against adversarial outcomes.
"""
import re

def normalize(s):
    return ' '.join(re.findall(r'[a-z0-9]+',str(s).lower().replace('_',' ').replace('-', ' ')))

def matches(label, prediction, mode='lexical_overlap', aliases=None):
    a=normalize(label)
    b=normalize(prediction)
    if not a or not b: return False
    options=[a]+[normalize(v) for v in (aliases or {}).get(label,[]) if normalize(v)]
    if mode=='exact': return any(s==b for s in options)
    if mode!='lexical_overlap': raise ValueError(f'Unknown matching rule: {mode}')
    for s in options:
        if s in b or b in s: return True
        if set(s.split())&set(b.split()): return True
    return False

def topk_correct(label, labels, k, *, mode='lexical_overlap', aliases=None):
    if k<=0: raise ValueError('k must be positive')
    return any(matches(label,s,mode,aliases) for s in labels[:k])
