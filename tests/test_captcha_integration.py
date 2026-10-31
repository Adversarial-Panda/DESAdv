"""Pure local CSV/manifest tests, no model weights or browser required."""
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def get_script(name):
    path=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location(name.removesuffix('.py'),path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_run(folder, rows, name):
    path=folder/name/'results.csv'
    path.parent.mkdir(parents=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','captcha_type','tag','round','success','error'])
        w.writeheader();w.writerows(rows)
    return path


def test_262_labels_available():
    c=get_script('check_captcha_data.py')
    result=c.audit(ROOT/'experiments/captcha')
    assert result['labeled_images']==262
    assert len(result['classes'])==10
    assert all(x['status']=='not_provided' for x in result['conditions'].values())


def test_csr_summarizer_denominator_and_conditions(tmp_path):
    m=get_script('captcha_report.py')
    rows=[dict(model='CLIP ViT-B/32',captcha_type='grid',tag='adv',round=str(i),
               success='True' if i==1 else 'False',error='') for i in range(1,4)]
    path=write_run(tmp_path,rows,'clip_run')
    result=m.load_run(path)
    assert result['csr_pct']==round(100/3,4)
    assert result['epsilon']=='8/255' and result['layout']=='Grid'
    assert result['attempted']==3 and result['errors']==0
    assert len(m.collect_runs(tmp_path,expected=3,strict=True))==1


def test_errors_are_not_removed_and_strict_rejects(tmp_path):
    m=get_script('captcha_report.py')
    rows=[dict(model='owlv2',captcha_type='area',tag='adv2',round='1',success='True',error=''),
          dict(model='owlv2',captcha_type='area',tag='adv2',round='2',success='',error='timeout')]
    write_run(tmp_path,rows,'owl_run')
    r=m.collect_runs(tmp_path,expected=2)[0]
    assert (r['completed'],r['errors'],r['attempted'],r['csr_pct']) == (1,1,2,100.0)
    import pytest
    with pytest.raises(ValueError,match='Cannot report'):
        m.collect_runs(tmp_path,expected=2,strict=True)


def test_duplicate_round_and_mixed_condition_rejected(tmp_path):
    m=get_script('captcha_report.py')
    rows=[dict(model='clip',captcha_type='grid',tag='benign',round='1',success='True',error=''),
          dict(model='clip',captcha_type='grid',tag='benign',round='1',success='False',error='')]
    path=write_run(tmp_path,rows,'wrong')
    import pytest
    with pytest.raises(ValueError,match='Duplicate'):
        m.load_run(path)


def test_manifest_mismatch_is_detected(tmp_path):
    mod=get_script('check_captcha_data.py')
    folder=tmp_path/'dataset'
    folder.mkdir()
    (folder/'captcha_images_labels.csv').write_text('image_name,label\na.jpeg,bee\nb.jpeg,ant\n')
    (folder/'captcha_images').mkdir()
    (folder/'captcha_images'/'a.jpeg').write_bytes(b'not real image')
    import pytest
    with pytest.raises(ValueError,match='mismatch'):
        mod.audit(tmp_path)
