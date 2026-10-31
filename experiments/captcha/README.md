# DES-Adv: Custom CAPTCHA evaluation

This module is the self-hosted Gradio CAPTCHA experiment used in **Table 6** of
*One Ensemble Does Not Fit All: Rethinking Query-Free Attacks on Web Vision Services*.
All browser automation here is restricted to the local Gradio test environment.
No commercial CAPTCHA challenge or account/session bypass is tested.

## Study design

- **262 images / 10 classes** are indexed by `dataset/captcha_images_labels.csv`.
- **Grid:** 3 × 3 selection, 2–4 target-class images.
- **Collage (source CLI `area`):** 6–8 scattered image crops with 2–3 targets.
- **Image conditions:** `benign`, `adv` (paper ε = 8/255), `adv2` (paper ε = 16/255).
- **500 completed sessions per solver/layout/condition** are required for manuscript comparability.
- **CSR** = 100 × successful complete sessions / completed sessions, reporting actual counts.
- Solvers include CLIP ViT-B/32 (grid), OWLv2 (collage), YOLOv8, Qwen2.5-VL,
  LLaVA, and Halligan/GPT-4o, with configurations recorded per run.

**Important:** Source tags are arbitrary names. `adv` and `adv2` represent those
ε values only after the image manifests, pixel scale, perturbation bounds,
and matching filenames have been verified. Model inference is distinct from
adversarial image generation; source weights and images are not in this archive.

## Local installation

Main DES-Adv package: `pip install -e '.[test]'` from repository root.
For the CAPTCHA lab, use a separate Python virtual environment for the provided
`requirements.reference.txt` (the original pinned environment; check current
availability/compatibility on your GPU). Python 3.12 or newer and CUDA are useful
for the large backends. Install Playwright Chromium in that environment.

```bash
cd experiments/captcha
python -m pip install -r requirements.reference.txt
python -m playwright install chromium
python download_weights.py --list
python download_weights.py --only clip owlv2 yolov8-cls
```

Images are intentionally not packaged. Place the **same filenames** in:

```text
dataset/
├── captcha_images_labels.csv   # included: 262 labeled filenames
├── captcha_images/             # benign data (not distributed)
├── captcha_images_adv/         # ε=8/255 data (not distributed)
└── CAPTCHA_ADV_IMAGES_2/       # ε=16/255 data (not distributed)
```

Check source file names/labels before evaluating:

```bash
python ../../scripts/check_captcha_data.py --root .
```

## Run controlled tests

From `experiments/captcha`:

```bash
# Terminal 1: host the custom challenge (one image condition at a time)
python captcha_gradio_app.py --type grid --images benign --port 7860

# Terminal 2: solve THIS local app (cannot run against an external hostname)
python captcha_solver.py --model clip --tag benign --rounds 500 --fast --headless --close-on-finish

# Repeat with --images adv / --tag adv and --images adv2 / --tag adv2;
# use --type area and --model owlv2 for the collage experiment.
python compare_results.py --details
```

For Halligan, use the separate environment defined in `halligan/pyproject.toml`
and run `halligan/run_gradio.py` on the local app. Halligan's components are
included with their MIT license; GPU dependencies and pretrained weights must
be installed separately. See `halligan/halligan/README.md` for the upstream
algorithm, and `halligan/run_gradio.py --help` for the local adapter.

## Generate manuscript-ready output

From the **repository root**:

```bash
python scripts/captcha_report.py --runs experiments/captcha/solver_results \
  --output outputs/captcha/table6.csv --expected 500 --strict
```

The reporter selects the most recent complete run for each solver/condition,
reports errors separately and **fails** in `--strict` mode if there are not
exactly 500 *completed* sessions. It does not silently drop failed requests
or invent results. Save manifests, logs and checkpoint hashes alongside
released table results for independent verification.

The original `compare_results.py` offers additional per-class/tile statistics.
The validated reporter is recommended for manuscript CSR because it checks
sample counts rather than silently excluding incomplete rounds.

## Core files

| File | Purpose |
| --- | --- |
| `captcha_gradio_app.py` | Grid/collage Gradio challenge and verdict logs |
| `area_captcha.py` | Collage creation and localization matching |
| `captcha_solver.py` | Local Playwright solver with multiple vision backends |
| `solver_backends.py` | CLIP, YOLOv8, OWLv2, Qwen2.5-VL, LLaVA and optional models |
| `halligan/run_gradio.py` | Self-hosted GPT-4o Halligan evaluation |
| `download_weights.py` | Pinned model download definitions; weights excluded |
| `compare_results.py` | Per-run and per-class metrics |
| `scripts/captcha_report.py` | Paper CSR with denominator/error audits |
| `scripts/check_captcha_data.py` | Exact filename pairing checks |

The two single-model wrappers (`captcha_clip_solver.py`, `captcha_yolo_solver.py`)
are retained as source-provided helper entrypoints.
