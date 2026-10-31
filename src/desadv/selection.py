"""Canonical Eq. (2)-(10) selection scoring; pure NumPy and unit-testable.

Raw scores: columns C (accuracy), G (1-mean inter-model cosine including self),
S (clean/noisy cosine, with the SAME clean predicted class).  NOT instability.
"""
import numpy as np

BASE = np.array([0.4, 0.5, 0.1], dtype=np.float64)

def normalized_signals(raw):
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError('Expected finite [n_models, 3] C/G/S signals')
    if np.any(a[:,0] < -1e-6) or np.any(a[:,0] > 1+1e-6):
        raise ValueError('C must be an accuracy fraction')
    if np.any(a[:,1] < -1e-6) or np.any(a[:,1] > 2+1e-6) or np.any(a[:,2] < -1-1e-6) or np.any(a[:,2] > 1+1e-6):
        raise ValueError('G must be in [0,2]; S must be in [-1,1]')
    return np.stack([np.clip(a[:,0],0,1),np.clip(a[:,1]/2,0,1),
                     np.clip((1+a[:,2])/2,0,1)], axis=1)

def adaptive_coefficients(q, lambda_=0.5, base=BASE):
    q=np.asarray(q,dtype=np.float64)
    base=np.asarray(base,dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('Expected finite [n_models,3] normalized signals')
    if base.shape != (3,) or np.any(base < 0) or base.sum() <= 0:
        raise ValueError('Invalid nonnegative three-signal base coefficients')
    if not (0 <= lambda_ <= 1):
        raise ValueError('lambda must be in [0,1]')
    spread=np.ptp(q,axis=0)
    r=spread/spread.sum() if spread.sum()>0 else np.zeros(3)
    a=base*(1+lambda_*r)
    return a/a.sum()

def select(raw, n_select=4, mode='C+G+S', lambda_=0.5, seed=42,
           sample_id=0, fixed_indices=(0,1,2,3)):
    q=normalized_signals(raw)
    m=len(q)
    if not (1 <= n_select <= m):
        raise ValueError('Invalid n_select')
    if mode=='fixed':
        ids=tuple(fixed_indices)
        if len(ids)!=n_select or len(set(ids))!=n_select or any(i<0 or i>=m for i in ids):
            raise ValueError('Fixed-set indices invalid')
        return np.asarray(ids,dtype=int),None
    if mode=='random':
        ids=np.random.default_rng(seed+sample_id).permutation(m)[:n_select]
        return ids,None
    if mode in ('C+G+S','C+G+S+dynamic'):
        coeffs=adaptive_coefficients(q, lambda_=lambda_)
    else:
        active=mode.replace('+fixed','').split('+')
        if not active or any(k not in ('C','G','S') for k in active):
            raise ValueError('Unknown selection variant')
        coeffs=np.array([BASE[i] if name in active else 0. for i,name in enumerate(('C','G','S'))])
        coeffs=coeffs/coeffs.sum()
    score=q@coeffs
    indices=np.lexsort((np.arange(m),-score))[:n_select]
    return indices,coeffs
