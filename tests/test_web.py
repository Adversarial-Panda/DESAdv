import json
import pytest
from desadv.web.matching import matches
from desadv.web.evaluate import evaluate
from desadv.web.providers import parse_top5
from desadv.web.predict import read_manifest

def test_matching():
    assert matches('tabby','tabby cat')
    assert not matches('goldfinch','bird')
    assert not matches('tabby','tabby cat',mode='exact')

def test_paired_topk_conditional():
    rows=[]
    for i,c,a in [('1',['tabby','animal'],['dog','animal']),('2',['dog','tabby'],['cow','dog']),('3',['dog'],['dog'])]:
        for split,pred in [('clean',c),('adv',a)]:
            rows.append(dict(provider='aws',method='SMER',image_id=i,split=split,label='tabby',predictions=pred))
    results=evaluate(rows,(1,2))
    assert results[0]['benign_correct']==1
    assert results[0]['conditional_asr_pct']==100
    assert results[1]['benign_correct']==2
    assert results[1]['conditional_asr_pct']==100

def test_unpaired_rejected():
    with pytest.raises(ValueError,match='Unpaired'):
        evaluate([dict(provider='aws',method='m',image_id='1',split='clean',label='cat',predictions=['cat'])])

def test_parser_and_manifest(tmp_path):
    assert parse_top5(json.dumps({'top5':['a','b','c','d','e']}))==['a','b','c','d','e']
    with pytest.raises(ValueError):parse_top5(json.dumps({'top5':['a','b']}))
    p=tmp_path/'a.csv';p.write_text('image_id,method,split,label,path\n1,A,clean,cat,img.png\n')
    with pytest.raises(ValueError):read_manifest(p)
