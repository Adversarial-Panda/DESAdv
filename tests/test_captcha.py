import pytest
from desadv.captcha.evaluate import csr

def test_grouped_csr():
    rows=[dict(solver='clip',layout='grid',condition=c,session_id=str(i),solved=(i<success))
          for c,success in [('clean',3),('adv',1)] for i in range(4)]
    x=csr(rows)
    assert [r['csr_pct'] for r in x]==[25.,75.]
    with pytest.raises(ValueError):csr([rows[0],rows[0]])
