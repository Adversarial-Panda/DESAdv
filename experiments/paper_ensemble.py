"""DES-Adv surrogate selection and ensemble attack evaluation.

Surrogate membership is selected before attack optimization. Each attack
preserves its own internal gradient aggregation and update procedure.

Usage:
  python experiments/paper_ensemble.py attack --help
  python experiments/paper_ensemble.py evaluate --help
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from paper_imagenet import (
    ALL_VARIANTS, QueryFileList, datasets_from_args, device_of, json_save,
    list_models, make_surrogate, make_target, select_models, set_seed, torch_load,
    save_tensor, _cache_identity,
)
import ensemble_attacks as attacks

# Attack implementations are defined in experiments/ensemble_attacks.py.
METHODS = ("ens", "svre", "adaea", "smer")
OPTIMIZATIONS = ("mifgsm", "pgd")


def _logits(out):
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def run_ensemble_attack(method, opt, models, x, y, *, eps, alpha, steps, momentum,
                    svre_sample_k=2, svre_refresh=5, adaea_beta=10.0,
                    adaea_threshold=-0.3, pgd_random_start=False):
    """Run an ensemble attack without modifying its internal aggregation.

    `eps` and `alpha` are floats on 0..1 normalized pixel range.
    All CLI attacks use ONE image per batch, because the original Ens.Attack
    normalizes the full-batch gradient with a global L1 norm.
    """
    if method not in METHODS or opt not in OPTIMIZATIONS:
        raise ValueError(f"Unsupported method/optimization: {method}/{opt}")
    if len(models) < 2:
        raise ValueError("At least 2 selected models are required")
    if method == "ens":
        if opt == "mifgsm":
            return attacks.ensemble_mi_fgsm(models, x, y, eps=eps, alpha=alpha,
                                             iters=steps, decay=momentum)
        return attacks.ensemble_pgd(models, x, y, eps=eps, alpha=alpha,
                                     iters=steps, random_start=pgd_random_start)
    if method == "svre":
        kwargs = dict(eps=eps, alpha=alpha, iters=steps,
                      sample_k=min(len(models), svre_sample_k),
                      refresh=svre_refresh)
        if opt == "mifgsm":
            return attacks.ensemble_svre_mi_fgsm(models, x, y,
                                                  decay=momentum, **kwargs)
        return attacks.ensemble_svre_pgd(models, x, y,
                                          random_start=pgd_random_start, **kwargs)
    if method == "adaea":
        kwargs = dict(models=models, eps=eps, alpha=alpha, iters=steps,
                      device=x.device, beta=adaea_beta, threshold=adaea_threshold)
        if opt == "mifgsm":
            atk = attacks.AdaEA_MIFGSM(momentum=momentum, **kwargs)
        else:
            atk = attacks.AdaEA_PGD(random_start=pgd_random_start, **kwargs)
        return atk(x, y)
    # The supplied SMER takes eps/alpha in *pixel units*, and divides by 255.
    smer_args = SimpleNamespace(eps=eps * 255.0, alpha=alpha * 255.0,
                                 momentum=momentum)
    if opt == "mifgsm":
        return attacks.MI_FGSM_SMER(models, x.clone(), y, smer_args,
                                    num_iter=steps)
    return attacks.PGD_SMER(models, x.clone(), y, smer_args, num_iter=steps)


def variant_selection(raw, variant, n_select, image_idx, seed, fixed_indices):
    ids = select_models(raw, variant, n_select, image_idx, seed, fixed_indices)
    return ids.tolist(), [1.0 / n_select] * n_select  # records ONLY; attack owns weights


def _parse_names(raw, valid, what):
    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError(f"No {what} provided")
    unexpected = [name for name in names if name not in valid]
    if unexpected:
        raise ValueError(f"Unknown {what}: {unexpected}; available {valid}")
    return list(dict.fromkeys(names))


def _config(args, method, variant, pool, fixed):
    return dict(method=method, variant=variant, optimization=args.optimization,
                pool=pool, select_n=args.select_n, fixed_indices=fixed,
                eps=args.eps, alpha=args.step_size, steps=args.steps,
                momentum=args.momentum, seed=args.seed,
                ensemble_weights=args.ensemble_weights,
                score_schema="C_accuracy_G_diversity_S_stability_paper_v1",
                selection_coefficients=[0.4,0.5,0.1], dynamic_lambda=0.5,
                svre_sample_k=args.svre_sample_k, svre_refresh=args.svre_refresh,
                adaea_beta=args.adaea_beta, adaea_threshold=args.adaea_threshold,
                pgd_random_start=args.pgd_random_start,
                test_list_hash=hashlib.sha256(Path(args.test_list).read_bytes()).hexdigest(),
                note="DES-Adv selector with ensemble attack optimizer")


def _attacks_root(out, opt, method, variant, weighting="native"):
    return out / "baselines" / opt / method / weighting / variant


def _check_cached(meta, scores_cache, query, pool, args):
    if meta["test_paths"] != query.paths:
        raise ValueError("Test list/order changed after `prepare`; rerun prepare and scores")
    if scores_cache["pool"] != pool:
        raise ValueError("Score cache surrogate pool mismatch; rerun scores with matching pool")
    if meta["k"] != args.k or scores_cache["k"] != args.k:
        raise ValueError("Neighborhood-size mismatch; rerun prepare and scores")
    if scores_cache["sigma"] != args.sigma:
        raise ValueError("Gradient-stability noise sigma mismatch; rerun scores")
    expected = _cache_identity(args, meta, pool, int(scores_cache["score_count"]))
    if scores_cache.get("identity") != expected:
        raise ValueError("Source image order, gradient seed or other score-cache identity differs")


def attack(args):
    set_seed(args.seed)
    methods = _parse_names(args.methods, METHODS, "methods")
    variants = _parse_names(args.variants, [v.lower() for v in ALL_VARIANTS], "variants")
    # restore meaningful original spelling: original choices are case-sensitive.
    canonical = {v.lower(): v for v in ALL_VARIANTS}
    variants = [canonical[v] for v in variants]
    fixed_indices = [int(t) for t in args.fixed_indices.split(",")]
    if args.ensemble_weights != "native":
        raise ValueError("Paper-compatible run forbids selector-derived attack weighting")
    if args.select_n < 2:
        raise ValueError("Use at least 2 selected models with these ensemble attacks")
    if len(fixed_indices) != args.select_n:
        raise ValueError("--fixed-indices must contain exactly --select-n indices")
    if (args.select_n >= len(list_models(args))):
        print("WARNING: selecting entire pool yields no dynamic COMPOSITION")
    if args.start_index < 0 or args.limit < 0 or args.chunk_size < 1:
        raise ValueError("Invalid start-index/limit/chunk-size")

    out = Path(args.out_dir)
    meta = torch_load(out / "prepared.pt")
    score_cache = torch_load(out / "scores_paper_v1.pt")
    if score_cache.get("score_schema") != "C_accuracy_G_diversity_S_stability_paper_v1":
        raise ValueError("A non-paper gradient signal cache was supplied")
    _, query = datasets_from_args(args)
    pool = list_models(args)
    _check_cached(meta, score_cache, query, pool, args)
    scores = score_cache["scores"]
    start = args.start_index
    end = min(len(query), start + args.limit) if args.limit else len(query)
    if end <= start:
        raise ValueError("No samples in requested slice")
    if end > len(scores):
        raise ValueError(f"Only first {len(scores)} samples have scores; need up to {end}.")

    device = device_of(args)
    print("Loading surrogate pool:", pool)
    models = [make_surrogate(name, device) for name in pool]
    print("IMPORTANT: 'native' uses each provided baseline's internal weighting; "
          "the criterion scores affect COMPOSITION only.")

    for method in methods:
        for variant in variants:
            # Suppress misleading 'adaptive' variant that would not be applied.
            if variant in ("fixed_adaptive", "random_adaptive") and args.ensemble_weights != "criterion":
                raise ValueError(f"{variant} requires --ensemble-weights criterion (Ens.Attack only).")
            settings = _config(args, method, variant, pool, fixed_indices)
            vardir = _attacks_root(out, args.optimization, method, variant, args.ensemble_weights)
            vardir.mkdir(parents=True, exist_ok=True)
            manifest = vardir / "settings.json"
            if manifest.exists() and json.loads(manifest.read_text()) != settings:
                raise ValueError(f"Configuration conflict at {vardir}; use a different --out-dir")
            json_save(manifest, settings)
            print(f"Running {method}/{args.optimization}, variant={variant}, images=[{start},{end})")
            for first in tqdm(range(start, end, args.chunk_size), desc=f"{method}:{variant}"):
                last = min(end, first + args.chunk_size)
                chunk_path = vardir / f"chunk_{first:05d}.pt"
                if chunk_path.exists():
                    old = torch_load(chunk_path)
                    if old["indices"].tolist() != list(range(first, last)):
                        raise ValueError(f"Existing chunk index mismatch at {chunk_path}")
                    continue
                images = []
                choices = []
                selected_weights = []
                runtimes = []
                perturb_linf = []
                perturb_l1mean = []
                for j in range(first, last):
                    # Identical random sequences for the same sample across variants/methods.
                    set_seed(args.seed + j)
                    x, y, _ = query[j]
                    x = x.unsqueeze(0).to(device)
                    y = torch.tensor([y], device=device, dtype=torch.long)
                    ids, weights = variant_selection(scores[j], variant,
                        args.select_n, j, args.seed, fixed_indices)
                    chosen = [models[k] for k in ids]
                    t0 = time.perf_counter()
                    adv = run_ensemble_attack(method, args.optimization, chosen, x, y,
                        eps=args.eps, alpha=args.step_size, steps=args.steps,
                        momentum=args.momentum, svre_sample_k=args.svre_sample_k,
                        svre_refresh=args.svre_refresh, adaea_beta=args.adaea_beta,
                        adaea_threshold=args.adaea_threshold,
                        pgd_random_start=args.pgd_random_start)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    sec = time.perf_counter() - t0
                    adv = adv.detach()
                    if adv.shape != x.shape or not torch.isfinite(adv).all():
                        raise ValueError(f"Invalid adversarial output for image {j}")
                    maximum = (adv - x).abs().max().item()
                    if maximum > args.eps + 1e-5 or adv.min() < -1e-6 or adv.max() > 1.000001:
                        raise ValueError(f"Budget/range violation {method}/{variant}: image {j}, "
                                         f"max Linf={maximum}, eps={args.eps}; not silently projected")
                    images.append(adv[0].cpu())
                    choices.append(torch.tensor(ids, dtype=torch.long))
                    selected_weights.append(torch.tensor(weights, dtype=torch.float32))
                    runtimes.append(sec)
                    perturb_linf.append(maximum)
                    perturb_l1mean.append((adv - x).abs().mean().item())
                    del adv, chosen
                save_tensor(chunk_path, dict(
                    indices=torch.arange(first, last), images=torch.stack(images),
                    labels=meta["test_labels"][first:last],
                    selected_ids=torch.stack(choices),
                    # Criterion weights are RECORDS ONLY in native baseline runs.
                    criterion_weights=torch.stack(selected_weights),
                    runtime_seconds=torch.tensor(runtimes),
                    linf=torch.tensor(perturb_linf),
                    mean_absolute_pixel_diff=torch.tensor(perturb_l1mean),
                ))
    print(f"Saved attacks under {out / 'baselines' / args.optimization}")


def _gather_chunks(directory):
    paths = sorted(directory.glob("chunk_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No attack chunks found in {directory}")
    chunks = [torch_load(p) for p in paths]
    ids = torch.cat([c["indices"] for c in chunks]).long()
    if ids.numel() != ids.unique().numel():
        raise ValueError(f"Duplicate indices in {directory}")
    order = ids.argsort()
    adv = torch.cat([c["images"] for c in chunks])[order]
    duration = torch.cat([c["runtime_seconds"] for c in chunks])[order]
    linf = torch.cat([c["linf"] for c in chunks])[order]
    l1 = torch.cat([c["mean_absolute_pixel_diff"] for c in chunks])[order]
    return ids[order], adv, duration, linf, l1


def evaluate(args):
    set_seed(args.seed)
    methods = _parse_names(args.methods, METHODS, "methods")
    variants = _parse_names(args.variants, [v.lower() for v in ALL_VARIANTS], "variants")
    canonical = {v.lower(): v for v in ALL_VARIANTS}
    variants = [canonical[v] for v in variants]
    out = Path(args.out_dir)
    meta = torch_load(out / "prepared.pt")
    _, query = datasets_from_args(args)
    if meta["test_paths"] != query.paths:
        raise ValueError("Test list changed after prepare")
    device = device_of(args)
    summary = []
    sample_rows = []
    if not args.targets:
        raise ValueError("Explicit --targets required; 3-model demo must not be misreported as nine-target evaluation")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for target_name in targets:
        target = make_target(target_name, device)
        loader = DataLoader(query, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        clean_pred = []
        with torch.no_grad():
            for x, y, ix in tqdm(loader, desc=f"clean:{target_name}"):
                clean_pred.extend(target(x.to(device)).argmax(1).cpu().tolist())
        clean_pred = torch.tensor(clean_pred, dtype=torch.long)
        labels = meta["test_labels"]
        for method in methods:
            for variant in variants:
                vardir = _attacks_root(out, args.optimization, method, variant, args.ensemble_weights)
                ids, adv_images, duration, linf, l1 = _gather_chunks(vardir)
                if ids[0] < 0 or ids[-1] >= len(query):
                    raise ValueError(f"Out-of-range ids in {vardir}")
                # Do not combine nonidentical image sets for a paired comparison.
                prediction = []
                with torch.no_grad():
                    for v in adv_images.split(args.batch_size):
                        prediction.extend(target(v.to(device)).argmax(1).cpu().tolist())
                pred = torch.tensor(prediction)
                subset_labels = labels[ids]
                subset_clean_pred = clean_pred[ids]
                clean_good = subset_clean_pred == subset_labels
                attack_success = pred != subset_labels
                good_n = int(clean_good.sum())
                conditional_asr = (float((attack_success & clean_good).sum()) / good_n
                                    if good_n else float("nan"))
                row = dict(target=target_name, method=method,
                           optimization=args.optimization, weighting=args.ensemble_weights, variant=variant,
                           N=len(ids), clean_correct=good_n,
                           clean_accuracy=float(clean_good.float().mean()),
                           adv_accuracy=float((pred == subset_labels).float().mean()),
                           conditional_asr=conditional_asr,
                           clean_to_wrong=int((attack_success & clean_good).sum()),
                           mean_runtime_seconds=float(duration.mean()),
                           mean_linf=float(linf.mean()),
                           mean_abs_pixel_diff=float(l1.mean()))
                summary.append(row)
                print(row)
                for k, sample_id in enumerate(ids.tolist()):
                    sample_rows.append(dict(target=target_name, method=method,
                        optimization=args.optimization, weighting=args.ensemble_weights, variant=variant,
                        image_idx=sample_id, clean_correct=int(clean_good[k]),
                        adv_wrong=int(attack_success[k]),
                        prediction_changed=int(pred[k] != subset_clean_pred[k])))
        del target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dest = out / "baselines" / args.optimization
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / f"summary_{args.ensemble_weights}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    with (dest / f"per_sample_{args.ensemble_weights}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader();writer.writerows(sample_rows)
    print(f"Summary: {dest / f'summary_{args.ensemble_weights}.csv'}; paired outcomes: "
          f"{dest / f'per_sample_{args.ensemble_weights}.csv'}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("attack", "evaluate"))
    p.add_argument("--test-list", default=None)
    p.add_argument("--val-dir", default="dataset/imagenet_validation")
    p.add_argument("--out-dir", default="results/paper_run")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16,
                   help="For evaluation only. Attacks use batch size ONE to preserve baseline semantics")
    p.add_argument("--six-model-pool", action="store_true", default=True)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select-n", type=int, default=4)
    p.add_argument("--fixed-indices", default="0,1,2,3")
    p.add_argument("--methods", default="ens,svre,adaea,smer")
    p.add_argument("--optimization", choices=OPTIMIZATIONS, default="mifgsm")
    p.add_argument("--variants", default="fixed,random,C,G,S,C+G,C+S,G+S,C+G+S,C+G+S+fixed")
    p.add_argument("--ensemble-weights", choices=("native",), default="native",
                   help="Native weighting only: DES-Adv selects model membership, not attack weights")
    p.add_argument("--eps", type=float, default=8/255)
    p.add_argument("--step-size", type=float, default=2/255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--svre-sample-k", type=int, default=2)
    p.add_argument("--svre-refresh", type=int, default=5)
    p.add_argument("--adaea-beta", type=float, default=10.0)
    p.add_argument("--adaea-threshold", type=float, default=-0.3)
    p.add_argument("--pgd-random-start", action="store_true")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0=all from start-index")
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--targets", required=False, default=None,
                   help="Explicit comma-separated target model names; supply all nine for the paper")
    return p


def main():
    args = build_parser().parse_args()
    if args.command == "attack":
        attack(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
