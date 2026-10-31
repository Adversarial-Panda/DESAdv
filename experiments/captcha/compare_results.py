"""Compare solver models on benign vs adversarial CAPTCHAs.

Run: python compare_results.py                 # all models, newest run per model and image set
     python compare_results.py --details       # also per-target success for each model

Runs are found in solver_results/*/results.csv and grouped by their model and tag columns.
Tile-level metrics join each round's challenge_id with the app's
dataset/captcha_results_<tag>.csv log (runs recorded before challenge ids were logged show "-").
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

TAGS = ('benign', 'adv', 'adv2')


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def latest_runs(output):
    """{(model, tag): rows} using the newest results.csv for each pair."""
    runs = {}
    for path in sorted(Path(output).glob('*/results.csv'), key=lambda p: p.stat().st_mtime):
        rows = [r for r in read_rows(path) if r.get('tag') in TAGS and not r['error']]
        if rows:
            # Area-select runs are listed separately from grid runs of the same model.
            model = rows[0]['model'] + (' [area]' if rows[0].get('captcha_type') == 'area' else '')
            runs[model, rows[0]['tag']] = (path, rows)
    return runs


def app_log(path):
    return {r['challenge_id']: r for r in read_rows(path)} if Path(path).exists() else {}


def metrics(rows, log):
    successes = sum(r['success'] == 'True' for r in rows)
    times = [float(r['inference_seconds']) for r in rows if r['inference_seconds']]
    result = {'rounds': len(rows), 'success': successes / len(rows),
              'seconds': sum(times) / len(times) if times else 0}
    tp = fp = fn = tn = 0
    matched = [log[r['challenge_id']] for r in rows if r.get('challenge_id') in log]
    for entry in matched:
        correct = {int(x) for x in entry['correct_indices'].split(',') if x}
        selected = {int(x) for x in entry['selected_indices'].split(',') if x}
        for i in range(9):
            tp += i in correct and i in selected
            fp += i not in correct and i in selected
            fn += i in correct and i not in selected
            tn += i not in correct and i not in selected
    if matched:
        result.update(tile_acc=(tp + tn) / (tp + fp + fn + tn),
                      precision=tp / (tp + fp) if tp + fp else 0,
                      recall=tp / (tp + fn) if tp + fn else 0)
    return result


def pct(value):
    return '-' if value is None else f'{value:.1%}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default='solver_results')
    parser.add_argument('--log', default='dataset/captcha_results_{tag}.csv')
    parser.add_argument('--details', action='store_true', help='Per-target success for each model')
    args = parser.parse_args()

    runs = latest_runs(args.output)
    if not runs:
        raise SystemExit(f'No tagged runs in {args.output}; run captcha_solver.py --tag benign/adv/adv2.')
    logs = {tag: app_log(args.log.format(tag=tag)) for tag in TAGS}
    models = sorted({model for model, _ in runs})
    results = {key: metrics(rows, logs[key[1]]) for key, (_, rows) in runs.items()}

    header = (f'{"model":32s}{"images":>8s}{"rounds":>8s}{"success":>10s}{"drop":>10s}'
              f'{"tile acc":>10s}{"precision":>11s}{"recall":>9s}{"sec/round":>11s}')
    print(header + '\n' + '-' * len(header))
    for model in models:
        benign = results.get((model, 'benign'))
        for tag in TAGS:
            r = results.get((model, tag))
            if not r:
                continue
            drop = f'{(benign["success"] - r["success"]) * 100:.1f}pp' if benign and tag != 'benign' else ''
            print(f'{model:32s}{tag:>8s}{r["rounds"]:>8d}{pct(r["success"]):>10s}{drop:>10s}'
                  f'{pct(r.get("tile_acc")):>10s}{pct(r.get("precision")):>11s}'
                  f'{pct(r.get("recall")):>9s}{r["seconds"]:>11.3f}')

    print('\ndrop = benign success minus this image set\'s success (percentage points).')
    print('Runs used:')
    for (model, tag), (path, _) in sorted(runs.items()):
        print(f'  {model} [{tag}]: {path}')

    if args.details:
        for model in models:
            print(f'\n{model}\n{"target":16s}' + ''.join(f'{tag:>10s}' for tag in TAGS))
            counts = {tag: defaultdict(lambda: [0, 0]) for tag in TAGS}
            for tag in TAGS:
                for r in runs.get((model, tag), (None, []))[1]:
                    counts[tag][r['target']][0] += r['success'] == 'True'
                    counts[tag][r['target']][1] += 1
            for target in sorted(set().union(*counts.values())):
                cells = [f'{c[0]}/{c[1]}' if (c := counts[tag].get(target)) else '-' for tag in TAGS]
                print(f'{target:16s}' + ''.join(f'{c:>10s}' for c in cells))


if __name__ == '__main__':
    main()
