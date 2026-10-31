"""Paired bootstrap of conditional ASR improvements using saved per-example predictions.

Example: python paired_bootstrap.py --csv results/desadv/ablation_per_sample.csv \
    --baseline fixed --method C+G+S
"""
import argparse
import pandas as pd
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--csv', default='results/desadv/ablation_per_sample.csv')
p.add_argument('--baseline', default='fixed')
p.add_argument('--method', default='C+G+S')
p.add_argument('--bootstrap', type=int, default=5000)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()

df = pd.read_csv(a.csv)
for target, d in df.groupby('target'):
    b = d.loc[d.variant.eq(a.baseline)].set_index('image_idx').sort_index()
    m = d.loc[d.variant.eq(a.method)].set_index('image_idx').sort_index()
    if len(b) == 0 or len(m) == 0:
        print(f'{target}: missing baseline or method, skipping')
        continue
    shared = b.index.intersection(m.index)
    b = b.loc[shared]
    m = m.loc[shared]
    if not (b.clean_correct.to_numpy() == m.clean_correct.to_numpy()).all():
        raise ValueError('Clean target correctness differs across variants; cannot compare')
    eligible = b.clean_correct.to_numpy().astype(bool)
    if not eligible.any():
        print(f'{target}: no target-clean-correct cases')
        continue
    baseline = b.adv_wrong.to_numpy(dtype=float)[eligible]
    method = m.adv_wrong.to_numpy(dtype=float)[eligible]
    diffs = method - baseline
    rng = np.random.default_rng(a.seed)
    samples = rng.integers(0, len(diffs), size=(a.bootstrap, len(diffs)))
    draws = diffs[samples].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    print(f'{target} | N_eligible={len(diffs)} | baseline ASR={baseline.mean()*100:.2f}% | '
          f'{a.method} ASR={method.mean()*100:.2f}% | gain={diffs.mean()*100:+.2f} pp | '
          f'paired 95% bootstrap CI=[{low*100:+.2f}, {high*100:+.2f}] pp')
