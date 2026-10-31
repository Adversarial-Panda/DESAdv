"""DES-Adv ImageNet RoC precomputation and selection.

DES-Adv ONLY chooses an input-dependent subset of surrogate model IDs.
Selection scores are never converted into adversarial attack weights.
The included reference MI-FGSM attack averages surrogate input gradients
uniformly (equal-weight gradient aggregation), then applies MI-FGSM.
This is NOT an implementation of the original SVRE, AdaEA, or SMER optimizers.
To evaluate those attacks, pass only selected IDs to their own implementations.

Selection signal definitions (q_i = [C_i, G_i/2, (1 + S_i)/2]):
  C: ordinary accuracy on labeled region-of-competence neighbors
  G: 1 - mean pairwise COSINE between each model's predicted-label CE
     input gradient and ALL candidate model gradients (including self)
  S: cosine of clean versus Gaussian-noised input CE gradients,
     both computed using the SAME clean predicted label
The 'C+G+S+fixed' variant has fixed signal coefficients.
The '+dynamic' variants use per-image *signal* coefficient adaptation.
Neither form assigns weights to the selected models during optimization.

Primary commands:
  python experiments/paper_imagenet.py prepare --help
  python experiments/paper_imagenet.py scores --help
  python experiments/paper_ensemble.py attack --help
  python experiments/paper_ensemble.py evaluate --help

Requirements: pip install torch torchvision timm pillow tqdm numpy pandas
Optional: pip install faiss-cpu
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ImageNet candidate surrogate pool from the manuscript.
FOUR_MODELS = [
    "deit_tiny_patch16_224",
    "vit_tiny_patch16_224",
    "efficientnet_b0",
    "xcit_tiny_12_p8_224",
]
SIX_MODELS = ["resnet18", "inception_v3"] + FOUR_MODELS
DINO_NAME = "vit_base_patch16_224.dino"
COEFFICIENTS = {"C": 0.4, "G": 0.5, "S": 0.1}
SCORE_SCHEMA = "C_accuracy_G_diversity_S_stability_paper_v1"
SCORE_FILE = "scores_paper_v1.pt"
SCORE_PARTIAL_FILE = "scores_partial_paper_v1.pt"
COMPETENCE_FILE = "competence_paper_v1.pt"
# S is cosine stability, NOT gradient instability; manuscript Eq. (7)-(8).
# Paper-specified coefficients are 0.4/0.5/0.1, lambda=0.5 and sigma=0.01.
# Checkpoint identifiers and preprocessing must be pinned for each experiment.
DYNAMIC_VARIANTS = ("C+G+S", "C+G+S+dynamic")
# Adaptive attack weighting is intentionally absent. Fixed/random controls also
# run through the same unmodified, equal-gradient-average reference optimizer.
ALL_VARIANTS = ["fixed", "random", "C", "G", "S", "C+G", "C+S", "G+S", "C+G+S", "C+G+S+fixed", "C+G+S+dynamic"]


def device_of(args):
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_tensor(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def list_models(args):
    return SIX_MODELS if args.six_model_pool else FOUR_MODELS


def base_transform():
    # Exactly matches user's raw pixels: 224 x 224, FloatTensor in [0, 1].
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


class QueryFileList(Dataset):
    def __init__(self, file_list: str, class_to_idx: Dict[str, int], transform=None):
        self.paths = [str(Path(line.strip()).expanduser().resolve())
                      for line in Path(file_list).read_text().splitlines() if line.strip()]
        if len(self.paths) == 0:
            raise ValueError(f"Empty list: {file_list}")
        if len(self.paths) != len(set(self.paths)):
            raise ValueError("Duplicate paths in test image list")
        self.class_to_idx = class_to_idx
        self.transform = transform or base_transform()
        bad = sorted({Path(path).parent.name for path in self.paths} - set(class_to_idx))
        if bad:
            raise ValueError(f"Unknown ImageNet class directories in test list: {bad[:15]}")
        missing = [p for p in self.paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"Test images not found: {missing[:5]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        y = self.class_to_idx[Path(path).parent.name]
        return x, y, idx


def datasets_from_args(args):
    if not args.test_list:
        raise ValueError("Supply an explicitly verified joint-benign-correct --test-list; no hard-case list is assumed")
    val = datasets.ImageFolder(args.val_dir, transform=base_transform())
    if len(val.classes) != 1000:
        raise ValueError(
            f"Validation ImageFolder has {len(val.classes)} classes, not 1000. "
            "ImageNet pretrained classifier outputs require the correct GLOBAL ImageNet-1K "
            "index mapping, not per-partition ImageFolder indices. Provide a complete "
            "ImageNet-1K validation directory organized by 1000 WNID class folders."
        )
    if not all(str(k).startswith("n") for k in val.classes):
        print("WARNING: class folder names do not look like ImageNet WNIDs. "
              "Verify the mapping matches timm pretrained output indices.")
    query = QueryFileList(args.test_list, val.class_to_idx)
    val_paths = {str(Path(p).resolve()) for p, _ in val.samples}
    common = val_paths.intersection(query.paths)
    if common:
        raise ValueError(f"Data leakage: {len(common)} identical paths in validation and test. "
                         f"Example: {next(iter(common))}")
    print(f"Loaded validation={len(val)} images / {len(val.classes)} classes, "
          f"selected test={len(query)} images, shared paths=0.")
    print("NOTE: path separation does not catch byte-identical copies in different locations.")
    return val, query


class NormWrapped(torch.nn.Module):
    def __init__(self, base, mean, std):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return self.base((x - self.mean) / self.std)


def make_surrogate(model_name, device):
    import timm
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    # Preserve the user's explicit normalization routing (the original 'key' rule).
    if model_name == "inception_v3" or model_name.startswith("vit_") or model_name.startswith("bit_"):
        mean, std = (0.5,) * 3, (0.5,) * 3
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    wrapped = NormWrapped(model, mean, std).to(device).eval()
    wrapped.requires_grad_(False)
    return wrapped


def make_target(model_name, device, override_mean=None, override_std=None):
    """Use official timm preprocessing for target models, unless overridden.

    DO NOT use this default to compare against historical results unless the
    exact preprocessing and pretrained checkpoints are verified.
    """
    import timm
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    config = timm.data.resolve_model_data_config(model)
    mean = tuple(override_mean or config["mean"])
    std = tuple(override_std or config["std"])
    print(f"Target {model_name}: mean={mean}, std={std}, "
          f"training_input_size={config.get('input_size')}; using 224x224 test images")
    wrapped = NormWrapped(model, mean, std).to(device).eval()
    wrapped.requires_grad_(False)
    return wrapped


def make_dino(device):
    import timm
    try:
        model = timm.create_model(DINO_NAME, pretrained=True, num_classes=0)
    except Exception as exc:
        raise RuntimeError(f"Cannot load DINO '{DINO_NAME}'. Check timm version and model names.") from exc
    data_config = timm.data.resolve_model_data_config(model)
    print(f"DINO preprocessing mean={data_config['mean']}, std={data_config['std']}")
    wrapped = NormWrapped(model, data_config["mean"], data_config["std"]).to(device).eval()
    wrapped.requires_grad_(False)
    return wrapped


@torch.no_grad()
def image_embeddings(model, dataset, batch_size, device, workers, title):
    outputs = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    for batch in tqdm(loader, desc=title):
        x = batch[0].to(device)
        z = model(x)
        if isinstance(z, (tuple, list)):
            z = z[0]
        if z.ndim == 3:
            z = z[:, 0]  # CLS representation, if forward returns tokens
        if z.ndim != 2:
            raise ValueError(f"Unexpected DINO feature shape: {z.shape}")
        outputs.append(F.normalize(z.float(), dim=-1).cpu())
    return torch.cat(outputs)


def get_neighbors(val_features, test_features, k):
    if k > len(val_features):
        raise ValueError("k exceeds number of validation samples")
    try:
        import faiss
        index = faiss.IndexFlatIP(val_features.shape[1])
        index.add(val_features.numpy().astype("float32"))
        similarities, indexes = index.search(test_features.numpy().astype("float32"), k)
        return torch.from_numpy(indexes).long(), torch.from_numpy(similarities)
    except ImportError:
        # Exact search in chunks, no faiss dependency (slow for huge reference sets).
        ids = []
        sims = []
        for v in test_features.split(64):
            sim = v @ val_features.T
            a, b = sim.topk(k, dim=1)
            sims.append(a)
            ids.append(b)
        return torch.cat(ids), torch.cat(sims)


def prepare(args):
    set_seed(args.seed)
    out = Path(args.out_dir)
    val, query = datasets_from_args(args)
    device = device_of(args)
    dino = make_dino(device)
    val_features = image_embeddings(dino, val, args.batch_size, device, args.workers, "DINO validation")
    test_features = image_embeddings(dino, query, args.batch_size, device, args.workers, "DINO test")
    del dino
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    idx, sims = get_neighbors(val_features, test_features, args.k)
    y_val = torch.tensor(val.targets).long()
    y_test = torch.tensor([query.class_to_idx[Path(x).parent.name] for x in query.paths]).long()
    save_tensor(out / "prepared.pt", {
        "neighbors": idx, "neighbor_similarities": sims,
        "val_labels": y_val, "test_labels": y_test,
        "test_paths": query.paths, "val_paths": [str(Path(p).resolve()) for p, _ in val.samples],
        "mapping": val.class_to_idx, "k": args.k,
        "dino_name": DINO_NAME,
    })
    json_save(out / "config.json", {"dataset": "ImageNet-1K", "test_list": args.test_list,
        "validation_dir": args.val_dir, "surrogates": list_models(args),
        "seed": args.seed, "sigma": args.sigma, "k": args.k,
        "notes": "DES-Adv C/G/S surrogate ranking"})
    print(f"Prepared neighbors saved to {out / 'prepared.pt'}")


@torch.no_grad()
def predict_correct(model, loader, device):
    correct = []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x).argmax(1)
        correct.append(pred.eq(y).cpu())
    return torch.cat(correct)


def pseudo_label_input_grad(model, x, reference_label=None):
    """Cross-entropy input gradient using the MODEL's own clean pseudo-label.

    When reference_label is supplied (noise perturbation), the *same clean*
    label is reused. Do not backpropagate through label selection.
    """
    z = x.detach().requires_grad_(True)
    logits = model(z)
    label = logits.detach().argmax(dim=1) if reference_label is None else reference_label
    grad = torch.autograd.grad(F.cross_entropy(logits, label, reduction="sum"), z)[0]
    return grad.detach(), label.detach()


def cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1, eps=1e-8).item()


def _cache_identity(args, data, pool, n):
    """Cache identity forbids cross-protocol reuse, including different image lists."""
    image_paths = data["test_paths"][:n]
    return {
        "schema": SCORE_SCHEMA,
        "pool": list(pool), "k": args.k, "sigma": float(args.sigma),
        "noise_seed": args.seed, "n": n,
        "image_list_digest": hashlib.sha256("\n".join(image_paths).encode()).hexdigest(),
        "dino": data.get("dino_name", "unknown"),
        "label_rule": "predicted_label_of_each_surrogate_on_clean_input",
        "noisy_label_rule": "reuse_clean_pseudolabel",
        "diversity_rule": "one_minus_mean_pairwise_cosine_including_self",
    }


def compute_scores(args):
    """Compute [C,G,S] consistent with the revised Method section.

    Pass 1: cache ordinary RoC classification accuracies per surrogate (streaming
    one model at a time for the DSEL dataset).
    Pass 2: retain all model gradients for ONE image only, then evaluate their
    full pairwise cosine matrix. This avoids caching ~gigabytes of gradients.

    WARNING: unlike the old query-to-RoC G, this needs all six model gradients
    on the SAME image. Requires enough RAM/VRAM to load the candidate pool.
    """
    if args.score_save_every < 1:
        raise ValueError("--score-save-every must be positive")
    set_seed(args.seed)
    out = Path(args.out_dir)
    data = torch_load(out / "prepared.pt")
    val, query = datasets_from_args(args)
    if data["test_paths"] != query.paths:
        raise ValueError("Test-list ordering/paths changed after prepare. Re-run prepare.")
    if data["k"] != args.k:
        raise ValueError("k differs from prepared neighbors")
    if data.get("dino_name") != DINO_NAME:
        raise ValueError("DINO model differs from prepared neighbors")
    device = device_of(args)
    N = len(query) if args.limit == 0 else min(args.limit, len(query))
    pool = list_models(args)
    M = len(pool)
    identity = _cache_identity(args, data, pool, N)
    full_cache = out / SCORE_FILE
    if full_cache.exists():
        cached = torch_load(full_cache)
        if cached.get("identity") != identity:
            raise ValueError(f"{full_cache} has different scoring settings. Choose a new --out-dir or move that cache.")
        if cached.get("scores", torch.empty(0)).shape != (N, M, 3):
            raise ValueError("Full score cache has unexpected tensor shape")
        print(f"Using verified complete {SCORE_SCHEMA} cache: {full_cache}")
        return

    # Reuse competent predictions only when the identity matches exactly.
    competence_path = out / COMPETENCE_FILE
    competence = torch.zeros((N, M), dtype=torch.float32)
    finished_models = 0
    if competence_path.exists():
        prior = torch_load(competence_path)
        if prior.get("identity") != identity:
            raise ValueError("Old competence cache has mismatched dataset/settings; choose a new --out-dir")
        competence = prior["competence"]
        finished_models = int(prior["finished_models"])
        if competence.shape != (N, M) or not 0 <= finished_models <= M:
            raise ValueError("Corrupted competence cache")
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    for midx in range(finished_models, M):
        print(f"Computing local competence for surrogate {midx+1}/{M}: {pool[midx]}")
        model = make_surrogate(pool[midx], device)
        correctness = predict_correct(model, val_loader, device)
        competence[:, midx] = correctness[data["neighbors"][:N]].float().mean(1)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_tensor(competence_path, {"identity": identity, "competence": competence,
                                       "finished_models": midx+1})

    scores = torch.zeros((N, M, 3), dtype=torch.float32)
    scores[:, :, 0] = competence
    partial_path = out / SCORE_PARTIAL_FILE
    finished_images = 0
    if partial_path.exists():
        prior = torch_load(partial_path)
        if prior.get("identity") != identity:
            raise ValueError("Old score checkpoint is incompatible with this method/settings; use a new --out-dir")
        finished_images = int(prior["finished_images"])
        if prior["scores"].shape != scores.shape or not 0 <= finished_images <= N:
            raise ValueError("Corrupted score checkpoint")
        scores = prior["scores"]
        print(f"Resuming inter-surrogate gradient diversity at image {finished_images}/{N}")

    print("Loading candidate surrogate pool for per-image pairwise gradient diversity...")
    models = [make_surrogate(name, device) for name in pool]
    try:
        for q_idx in tqdm(range(finished_images, N), desc="C/G/S scoring"):
            x, _y, _ = query[q_idx]
            x = x.unsqueeze(0).to(device)
            rng = torch.Generator(device="cpu").manual_seed(args.seed + q_idx)
            noise = torch.randn(tuple(x.shape), generator=rng).to(device) * args.sigma
            noisy_x = (x + noise).clamp(0, 1)
            gradients = []
            stability = []
            for model in models:
                g_clean, pseudo = pseudo_label_input_grad(model, x)
                g_noisy, _ = pseudo_label_input_grad(model, noisy_x, pseudo)
                gradients.append(g_clean.flatten().cpu().float())
                stability.append(cos(g_clean, g_noisy))
            # Mean over ALL models, including the diagonal (self cosine).
            grad_vectors = torch.stack(gradients)
            unit_gradients = F.normalize(grad_vectors, p=2, dim=1, eps=1e-8)
            cosine_matrix = (unit_gradients @ unit_gradients.T).clamp(-1, 1)
            diversity = 1.0 - cosine_matrix.mean(dim=1)
            scores[q_idx, :, 1] = diversity
            scores[q_idx, :, 2] = torch.tensor(stability, dtype=torch.float32)
            if (q_idx + 1) % args.score_save_every == 0 or q_idx + 1 == N:
                save_tensor(partial_path, {"identity": identity, "scores": scores,
                                           "finished_images": q_idx + 1})
    finally:
        del models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_tensor(full_cache, {"identity": identity, "score_schema": SCORE_SCHEMA,
                              "scores": scores, "pool": pool, "sigma": args.sigma,
                              "k": args.k, "score_count": N})
    print(f"Saved local competence, gradient diversity and stability to {full_cache}")


def criterion_matrix(raw):
    """Paper Eq. (8): C, G/2, (1+S)/2 on [0,1]."""
    scores = torch.stack([
        raw[..., 0].clamp(0, 1),
        (raw[..., 1] / 2.0).clamp(0, 1),
        ((1.0 + raw[..., 2]) / 2.0).clamp(0, 1)
    ], dim=-1)
    return scores


def adaptive_signal_coefficients(normed: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Bounded, image-adaptive signal coefficients from Mx3 normalized scores.

    r_j = (max_i q_ij - min_i q_ij) / sum_h (max_i q_ih - min_i q_ih)
    a_j(x) = a0_j * (1 + strength * r_j) / sum_h a0_h * (1 + strength * r_h)

    A zero-spread image retains a0; strength=0 exactly reproduces a0.
    Discrimination is only a heuristic, not evidence that a signal is reliable.
    """
    if normed.ndim != 2 or normed.shape[1] != 3:
        raise ValueError(f"Expected normalized Mx3 signal tensor, got {tuple(normed.shape)}")
    if not (0.0 <= strength <= 1.0):
        raise ValueError("--dynamic-lambda must lie in [0, 1]")
    base = torch.tensor([COEFFICIENTS[c] for c in ["C", "G", "S"]],
                        dtype=normed.dtype, device=normed.device)
    spread = normed.max(dim=0).values - normed.min(dim=0).values
    denom = spread.sum()
    if strength == 0.0 or denom.item() <= 1e-8:
        return base / base.sum()
    importance = spread / denom
    modified = base * (1.0 + strength * importance)
    return modified / modified.sum()


def select_models(raw, variant, n_select, sample_id, seed, fixed_indices,
                  dynamic_lambda=0.5):
    """Paper Eq. (8)-(10): C, G and gradient *stability* S.

    The primary C+G+S uses adaptive range coefficients (lambda=0.5).
    C+G+S+fixed uses lambda=0; selection is performed only once per image.
    No attack-optimizer weights are set here. Top-k tie breaks use a stable index.
    """
    M = raw.shape[0]
    if not 1 <= n_select <= M:
        raise ValueError('Invalid subset size')
    if variant == 'fixed':
        if len(fixed_indices) != n_select or len(set(fixed_indices)) != n_select:
            raise ValueError('Fixed indices must be distinct and match subset size')
        if any(i < 0 or i >= M for i in fixed_indices):
            raise ValueError('Fixed indices out of range')
        return torch.tensor(fixed_indices, dtype=torch.long)
    if variant == 'random':
        generator = torch.Generator().manual_seed(seed * 1000003 + sample_id)
        return torch.randperm(M, generator=generator)[:n_select]
    if variant not in ALL_VARIANTS:
        raise ValueError(f'Unknown selector variant: {variant}')
    q = criterion_matrix(raw)
    if variant in ('C+G+S', 'C+G+S+dynamic'):
        a = adaptive_signal_coefficients(q, strength=dynamic_lambda)
    else:
        v = 'C+G+S' if variant == 'C+G+S+fixed' else variant
        chosen = v.split('+')
        a = torch.tensor([COEFFICIENTS[key] if key in chosen else 0.0
                          for key in ('C','G','S')], dtype=q.dtype, device=q.device)
        a = a / a.sum()
    scores = q @ a
    # Explicit deterministic tie ordering (first pool index wins ties).
    ranked = sorted(range(M), key=lambda i: (-float(scores[i]), i))
    return torch.tensor(ranked[:n_select], dtype=torch.long)


def mi_fgsm(x, y, models, ids, eps, step, iters, mu=1.0):
    """Reference untargeted MI-FGSM with UNIFORM surrogate-gradient averaging.

    DES-Adv passes only ids. This optimizer does not read selection scores or
    DES-Adv weights. Actual optimizer-specific weighting in SVRE/AdaEA/SMER
    must be implemented by those respective attacks, not here.
    """
    if ids.numel() == 0:
        raise ValueError("At least one surrogate must be selected")
    x_orig = x.detach()
    adv = x_orig.clone()
    momentum = torch.zeros_like(adv)
    y = y.to(adv.device)
    for _ in range(iters):
        total_grad = torch.zeros_like(adv)
        for model_idx in ids.tolist():
            # Independent surrogate backward pass; no DES-Adv weighting.
            z = adv.detach().requires_grad_(True)
            logits = models[model_idx](z)
            loss = F.cross_entropy(logits, y)
            g = torch.autograd.grad(loss, z, only_inputs=True)[0].detach()
            total_grad += g
        total_grad /= ids.numel()  # underlying attack's uniform aggregation
        avg_mag = total_grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        momentum = mu * momentum + total_grad / avg_mag
        adv = adv.detach() + step * momentum.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
        adv = adv.clamp(0.0, 1.0).detach()
    return adv


def attack(args):
    set_seed(args.seed)
    out = Path(args.out_dir)
    meta = torch_load(out / "prepared.pt")
    s = torch_load(out / SCORE_FILE)
    if s.get("score_schema") != SCORE_SCHEMA:
        raise ValueError("Score cache has a different gradient definition. Re-run scores with this code.")
    _, query = datasets_from_args(args)
    expected_identity = _cache_identity(args, meta, list_models(args), s["score_count"])
    if s.get("identity") != expected_identity:
        raise ValueError("Score-cache protocol does not match this run (pool, images, sigma, seed, or k). Re-run scores.")
    if s["pool"] != list_models(args):
        raise ValueError("Requested surrogate pool differs from score cache. Recompute scores.")
    if meta["test_paths"] != query.paths:
        raise ValueError("Changed test order; re-run prepare and scores.")
    scores = s["scores"]
    pool = s["pool"]
    fixed_indices = [int(v) for v in args.fixed_indices.split(",")]
    device = device_of(args)
    print("Loading all surrogate models for attack generation...")
    models = [make_surrogate(name, device) for name in pool]
    variants = args.variants.split(",")
    if not 0.0 <= args.dynamic_lambda <= 1.0:
        raise ValueError("--dynamic-lambda must lie in [0, 1]")
    for variant in variants:
        if variant not in ALL_VARIANTS:
            raise ValueError(f"Invalid variant '{variant}'. Available: {ALL_VARIANTS}. "
                             "Weighted variants were removed from this selection-only version.")
        vardir = out / args.attack_folder / variant
        vardir.mkdir(parents=True, exist_ok=True)
        manifest_path = vardir / "settings.json"
        settings = {"variant": variant, "pool": pool, "n_select": args.select_n,
                    "eps": args.eps, "alpha": args.step_size, "steps": args.steps,
                    "seed": args.seed, "fixed_indices": fixed_indices,
                    "coefficients": COEFFICIENTS,
                    "method_version": "paper_selector_v1_reference_only",
                    "score_schema": SCORE_SCHEMA, "k": args.k, "sigma": args.sigma,
                    "attack_aggregation": "uniform_mean_surrogate_input_gradients",
                    "selection_only": True,
                    "notes": "reference-only MI-FGSM; primary ensemble results must use paper_ensemble.py"}
        if variant in DYNAMIC_VARIANTS:
            settings["dynamic_lambda"] = args.dynamic_lambda
            settings["adaptive_rule"] = "range across six surrogates; normalized; base*(1+lambda*r)"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old != settings:
                raise ValueError(f"Existing attack settings differ in {vardir}; use new --out-dir")
        else:
            json_save(manifest_path, settings)
        print(f"Generating variant {variant}")
        chunk_size = args.chunk_size
        num = len(query) if args.limit == 0 else min(args.limit, len(query))
        if num > len(scores):
            raise ValueError(f"Only {len(scores)} samples have cached scores; requested {num}. Re-run scores with a larger --limit or 0")
        for first in tqdm(range(0, num, chunk_size), desc=variant):
            n = min(chunk_size, num - first)
            chunk_path = vardir / f"chunk_{first:05d}.pt"
            if chunk_path.exists():
                old = torch_load(chunk_path)
                if old["indices"].tolist() != list(range(first, first+n)):
                    raise ValueError(f"Corrupted chunk indexing: {chunk_path}")
                continue  # supports resuming
            attacked = []
            selected_ids = []
            adaptive_coeffs = []
            for j in range(first, first+n):
                x, y, _ = query[j]
                ids = select_models(scores[j], variant, args.select_n, j, args.seed,
                                    fixed_indices, dynamic_lambda=args.dynamic_lambda)
                if variant in DYNAMIC_VARIANTS:
                    # Persist coefficients to support auditing the adaptive mechanism.
                    adaptive_coeffs.append(adaptive_signal_coefficients(
                        criterion_matrix(scores[j]), strength=args.dynamic_lambda))
                adv = mi_fgsm(x.unsqueeze(0).to(device),
                               torch.tensor([y], dtype=torch.long, device=device),
                               models, ids, args.eps, args.step_size, args.steps)
                if not torch.isfinite(adv).all():
                    raise ValueError(f"Nonfinite adversarial image for sample {j}")
                if (adv - x.unsqueeze(0).to(device)).abs().max().item() > args.eps + 1e-6:
                    raise AssertionError("Perturbation budget exceeded")
                attacked.append(adv[0].cpu())
                selected_ids.append(ids)
            payload = {
                "indices": torch.arange(first, first+n),
                "images": torch.stack(attacked), "labels": meta["test_labels"][first:first+n],
                "selected_ids": torch.stack(selected_ids),
            }
            if adaptive_coeffs:
                payload["adaptive_signal_coefficients"] = torch.stack(adaptive_coeffs)
            save_tensor(chunk_path, payload)
    print(f"Finished attacks. Results under {out / args.attack_folder}/<variant>/chunk_*.pt")


def evaluate(args):
    set_seed(args.seed)
    out = Path(args.out_dir)
    meta = torch_load(out / "prepared.pt")
    _, query = datasets_from_args(args)
    if meta["test_paths"] != query.paths:
        raise ValueError("Changed test order; re-run prepare")
    device = device_of(args)
    variants = args.variants.split(",")
    results = []
    per_sample = []
    for target_name in args.targets.split(","):
        print(f"Evaluating unseen target: {target_name}")
        target = make_target(target_name, device)
        target.eval()
        clean_pred = []
        loader = DataLoader(query, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        with torch.no_grad():
            for x, y, ids in tqdm(loader, desc=f"clean {target_name}"):
                clean_pred.extend(target(x.to(device)).argmax(1).cpu().tolist())
        clean_pred = torch.tensor(clean_pred)
        labels = meta["test_labels"]
        clean_correct = clean_pred.eq(labels)
        for variant in variants:
            manifest_path = out / args.attack_folder / variant / "settings.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"No versioned settings for {variant}; refusing to evaluate unknown attack chunks")
            settings = json.loads(manifest_path.read_text())
            if (settings.get("method_version") != "paper_selector_v1_reference_only" or
                    settings.get("score_schema") != SCORE_SCHEMA or
                    not settings.get("selection_only")):
                raise ValueError(f"Incompatible attack settings for {variant}; refusing to mix historical results")
            files = sorted((out / args.attack_folder / variant).glob("chunk_*.pt"))
            if not files:
                raise FileNotFoundError(f"No adversarial chunks for variant={variant}")
            ids_all, preds_all = [], []
            with torch.no_grad():
                for file in tqdm(files, desc=f"{target_name} {variant}"):
                    chunk = torch_load(file)
                    pred = target(chunk["images"].to(device)).argmax(1).cpu()
                    ids_all.extend(chunk["indices"].tolist())
                    preds_all.extend(pred.tolist())
            if len(ids_all) != len(set(ids_all)):
                raise ValueError(f"Repeated test indices for {variant}")
            ids = torch.tensor(ids_all)
            if not torch.equal(ids.sort().values, torch.arange(len(ids))):
                raise ValueError(f"Missing indices or nonzero start for {variant}")
            order = torch.argsort(ids)
            adv_pred = torch.tensor(preds_all)[order]
            subset_correct = clean_correct[:len(ids)]
            subset_label = labels[:len(ids)]
            clean_n = int(subset_correct.sum())
            broken = (adv_pred != subset_label)
            conditional_asr = float((broken & subset_correct).sum() / clean_n) if clean_n else float("nan")
            adv_acc = float(adv_pred.eq(subset_label).float().mean())
            rows = {"target": target_name, "variant": variant, "N": len(ids),
                    "clean_correct": clean_n, "clean_accuracy": float(subset_correct.float().mean()),
                    "adv_accuracy": adv_acc, "conditional_asr": conditional_asr,
                    "prediction_change": float(adv_pred.ne(clean_pred[:len(ids)]).float().mean())}
            results.append(rows)
            print(rows)
            for j in range(len(ids)):
                per_sample.append({"target": target_name, "variant": variant, "image_idx": j,
                    "label": int(subset_label[j]), "clean_correct": int(subset_correct[j]),
                    "adv_wrong": int(broken[j]), "prediction_changed": int(adv_pred[j] != clean_pred[j])})
        del target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report_dir = out / args.attack_folder
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "ablation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with open(report_dir / "ablation_per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    print(f"Saved {report_dir / 'ablation_summary.csv'} and {report_dir / 'ablation_per_sample.csv'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "scores"])
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--val-dir", default="dataset/imagenet_validation")
    parser.add_argument("--out-dir", default="results/paper_run")
    parser.add_argument("--attack-folder", default="attacks_selection_only_diversity_v2",
                        help="Separate new gradient-diversity outputs from earlier/weighted attack chunks")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--six-model-pool", action="store_true", default=True,
                        help="Enable ResNet18+InceptionV3+the provided 4 models (6 in total)")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Paper Gaussian noise sigma")
    parser.add_argument("--score-save-every", type=int, default=25,
                        help="Checkpoint per-image G/S score calculation every N images")
    parser.add_argument("--dynamic-lambda", type=float, default=0.5,
                        help="Paper coefficient adaptation lambda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-n", type=int, default=4,
                        help="Paper selects four from the six candidate models")
    parser.add_argument("--fixed-indices", default="0,1,2,3",
                        help="Index positions in the pool for prespecified fixed subset")
    parser.add_argument("--variants", default="fixed,random,C,G,S,C+G,C+S,G+S,C+G+S,C+G+S+fixed",
                        help="Comma-separated selection variants; no DES attack-weight variants")
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--step-size", type=float, default=2/255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0,
                        help="0=all images; use e.g. 10 for short scores/attack runs")
    parser.add_argument("--targets", default="resnet152,vgg19,vit_base_patch16_224")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        compute_scores(args)


if __name__ == "__main__":
    main()
