#!/usr/bin/env python3
"""Summarize the original self-hosted Gradio CAPTCHA solver_results CSV logs.

Table 6 CSR = successful sessions / completed sessions. A failed or incomplete
attempt is separately visible; it cannot silently disappear from validation.
Use --strict with --expected 500 before claiming manuscript comparability.
No browser or remote-service interaction occurs in this script.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CONDITIONS = {'benign': 'benign', 'adv': '8/255', 'adv2': '16/255'}
LAYOUTS = {'grid': 'Grid', 'area': 'Collage'}


def parse_success(raw):
    value = str(raw).strip().lower()
    if value in {'true', '1'}:
        return True
    if value in {'false', '0'}:
        return False
    return None


def load_run(path):
    """Return one complete run; report invalid states rather than dropping rows."""
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No logged attempts: {path}')
    for key in ('model', 'captcha_type', 'tag', 'success', 'error', 'round'):
        if key not in rows[0]:
            raise ValueError(f'Missing {key!r}: {path}')
    keys = {(r['model'].strip(), r['captcha_type'].strip(), r['tag'].strip()) for r in rows}
    if len(keys) != 1:
        raise ValueError(f'Mixed model/layout/tag in one run: {path}')
    model, layout, tag = next(iter(keys))
    if layout not in LAYOUTS or tag not in CONDITIONS:
        raise ValueError(f'Unknown CAPTCHA type or image condition: {layout}, {tag} in {path}')
    round_ids = [r['round'].strip() for r in rows]
    if len(set(round_ids)) != len(round_ids) or not all(x.isdigit() and int(x) >= 1 for x in round_ids):
        raise ValueError(f'Duplicate or invalid round indices: {path}')
    successes = completed = failed = 0
    for r in rows:
        err = bool(r['error'].strip())
        status = parse_success(r['success'])
        if err or status is None:
            failed += 1
            if status is not None and err:
                raise ValueError(f'Both success and error are set: {path} round={r["round"]}')
        else:
            completed += 1
            successes += int(status)
    return {
        'solver': model,
        'layout': LAYOUTS[layout],
        'condition': tag,
        'epsilon': CONDITIONS[tag],
        'attempted': len(rows),
        'completed': completed,
        'errors': failed,
        'successes': successes,
        'csr_pct': round(100 * successes / completed, 4) if completed else None,
        'run_csv': str(path),
    }


def collect_runs(folder, expected=500, strict=False):
    candidates = {}
    for path in sorted(folder.glob('*/results.csv'), key=lambda p: (p.stat().st_mtime_ns, str(p))):
        result = load_run(path)
        key = (result['solver'], result['layout'], result['condition'])
        # Keep the latest run for the same key, without merging unrelated sessions.
        candidates[key] = result
    if not candidates:
        raise ValueError(f'No results.csv files found inside {folder}/<run>/')
    results = [candidates[k] for k in sorted(candidates)]
    if strict:
        incomplete = [f'{r["solver"]}/{r["layout"]}/{r["condition"]}: '
                      f'{r["completed"]} completed, {r["errors"]} errors '
                      f'({r["attempted"]} attempted)'
                      for r in results if r['completed'] != expected or r['errors']]
        if incomplete:
            raise ValueError(f'Cannot report {expected}-session paper result:\n'+'\n'.join(incomplete))
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('experiments/captcha/solver_results'))
    p.add_argument('--output', type=Path, default=Path('outputs/captcha/table6.csv'))
    p.add_argument('--expected', type=int, default=500)
    p.add_argument('--strict', action='store_true', help='Require exactly N completed sessions, zero errors')
    args=p.parse_args(argv)
    if args.expected < 1:
        p.error('--expected must be positive')
    rows=collect_runs(args.runs, args.expected, args.strict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f'CSV: {args.output}')

if __name__=='__main__':
    main()
