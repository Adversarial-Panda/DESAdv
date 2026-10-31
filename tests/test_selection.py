import numpy as np
import pytest
from desadv.selection import normalized_signals, adaptive_coefficients, select
from desadv.fixed_sets import FIXED_SETS

def test_stability_not_instability():
    q=normalized_signals(np.array([[1,0,1],[0,2,-1]]))
    np.testing.assert_allclose(q,[[1,0,1],[0,1,0]])
    assert select([[1,0,1],[0,2,-1]],1,'S')[0].tolist()==[0]

def test_adaptive_coefficient_spread_and_ties():
    q=np.full((6,3),.5)
    np.testing.assert_allclose(adaptive_coefficients(q),[.4,.5,.1])
    q[0,0]=.9
    assert adaptive_coefficients(q,.5)[0]>.4
    assert select(np.tile([.5,.3,.4],(6,1)),4)[0].tolist()==[0,1,2,3]

def test_fixed_sets_and_invalid_data():
    assert len(FIXED_SETS)==15
    assert len({frozenset(v) for v in FIXED_SETS.values()})==15
    assert FIXED_SETS[3]==(1,2,5,4)
    with pytest.raises(ValueError):normalized_signals([[.5,.2,1.4]])
    with pytest.raises(ValueError):normalized_signals([[.5,.2,float('nan')]])

def test_real_pytorch_selector_matches_math():
    import torch
    from paper_imagenet import select_models
    rng=np.random.default_rng(100)
    raw=np.stack([rng.uniform(0,1,6),rng.uniform(0,2,6),rng.uniform(-1,1,6)],axis=-1)
    for v in ('C','G','S','C+G','C+S','G+S','C+G+S','C+G+S+fixed'):
        a,_=select(raw,4,mode=v)
        b=select_models(torch.tensor(raw,dtype=torch.float32),v,4,0,42,[0,1,2,3])
        assert a.tolist()==b.tolist()
