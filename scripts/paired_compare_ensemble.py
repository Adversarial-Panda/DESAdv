"""Paired image-level bootstrap comparison for DES-Adv ensemble attacks.

Example:
python paired_compare_ensemble.py \
  --csv results/desadv_six/baselines/mifgsm/per_sample_native.csv \
  --ensemble-method ens --baseline fixed --variant C+G+S
"""
import argparse
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--csv', required=True)
p.add_argument('--ensemble-method', choices=['ens', 'svre', 'adaea', 'smer'], required=True)
p.add_argument('--baseline', default='fixed')
p.add_argument('--variant', default='C+G+S')
p.add_argument('--bootstrap', type=int, default=5000)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()

df = pd.read_csv(a.csv)
df = df.loc[df.method.eq(a.ensemble_method)].copy()
if df.empty:
    raise SystemExit(f'No records for method={a.ensemble_method}')
for target, data in df.groupby('target'):
    b = data.loc[data.variant.eq(a.baseline)].set_index('image_idx')
    v = data.loc[data.variant.eq(a.variant)].set_index('image_idx')
    if b.empty or v.empty:
        print(f'{target}: missing one or both variants; skip')
        continue
    if not b.index.is_unique or not v.index.is_unique:
        raise ValueError('Duplicated image_idx within variant; inspect CSV')
    shared = b.index.intersection(v.index).sort_values()
    if len(shared) != len(b) or len(shared) != len(v):
        raise ValueError(f'{target}: methods tested different image subsets; rerun to align subsets')
    b, v = b.loc[shared], v.loc[shared]
    if not np.array_equal(b.clean_correct.to_numpy(), v.clean_correct.to_numpy()):
        raise ValueError('Clean target correctness differs for same image')
    eligible = b.clean_correct.to_numpy().astype(bool)
    if eligible.sum() == 0:
        print(f'{target}: no clean-correct images; skip')
        continue
    b_success = b.adv_wrong.to_numpy()[eligible].astype(float)
    v_success = v.adv_wrong.to_numpy()[eligible].astype(float)
    differences = v_success - b_success
    rng = np.random.default_rng(a.seed)
    draws = []
    for _ in range(a.bootstrap):
        chosen = rng.integers(0, len(differences), len(differences))
        draws.append(differences[chosen].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    print(f'{target}: method={a.ensemble_method}, n_clean_correct={len(differences)}, '
          f'{a.baseline}={b_success.mean()*100:.2f}% '
          f'{a.variant}={v_success.mean()*100:.2f}% '
          f'gain={differences.mean()*100:+.2f} percentage points '
          f'paired bootstrap 95% CI [{low*100:+.2f}, {high*100:+.2f}] pp')
