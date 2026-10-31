# DES-Adv

**One Ensemble Does Not Fit All: Rethinking Query-Free Attacks on Web Vision Services**  

DES-Adv is an input-adaptive surrogate selection framework for **query-free
black-box transfer attacks**. For each input image, it retrieves a local region
of competence (RoC), ranks candidate surrogates using local competence (C),
gradient diversity (G), and gradient stability (S), and selects an ensemble
before attack generation. The target is **not queried during selection or
optimization**. Target predictions are reserved for offline test-set eligibility
checks and final evaluation.

## Repository layout

```text
DES-Adv/
├── configs/                  ImageNet and Web evaluation configurations
├── src/desadv/
│   ├── selection.py          C/G/S ranking and adaptive coefficients
│   ├── fixed_sets.py         Exhaustive 4-of-6 ensemble configurations
│   ├── web/                  External-service adapters and ASR evaluation
│   └── captcha/              Local solver utilities and CSR evaluation
├── experiments/
│   ├── paper_imagenet.py     ImageNet RoC and score computation
│   ├── paper_ensemble.py     Attack generation and target evaluation
│   ├── ensemble_attacks.py   EnsAttack, SVRE, AdaEA, SMER
│   ├── captcha/              Gradio CAPTCHA suite, vision backends, Halligan adapter
│   └── cifar/                CIFAR experiment requirements
├── notebooks/web/            Service-specific evaluation notebooks
├── scripts/                  Data preparation, fixed sets, and statistics
├── tests/                    Automated checks
├── docs/                     Protocols and manuscript-to-code mapping
└── .github/workflows/ci.yml  Test workflow
```

## Installation

Python 3.10+ is recommended. Install PyTorch with the CUDA build appropriate
for your environment, then:

```bash
python -m pip install -e '.[vision,test]'
python -m pytest -q
```

For Web services and local CAPTCHA evaluation, optional dependencies are
available as `.[web]` and `.[captcha]`. Credentials are read from environment
variables; no credentials or model weights are included in this repository.

## Manuscript settings

| Component | Configuration |
| --- | --- |
| Datasets | ImageNet-1K; CIFAR-10/100 |
| ImageNet DSEL | 30,000 labeled images |
| ImageNet evaluation | 1,000 images jointly benign-correct on the unseen targets |
| Representation | DINO ViT-B/16 with L2-normalized embeddings |
| RoC search | FAISS exact cosine similarity, k=7 |
| Ensemble | Six candidate surrogates; four selected per input |
| Signals | C: local accuracy; G: diversity; S: gradient stability |
| Normalization | `q_i = (C_i, G_i/2, (1+S_i)/2)` |
| Coefficients | `a0 = (0.4, 0.5, 0.1)`; adaptation `lambda=0.5` |
| Noise standard deviation | `sigma=0.01` |
| Attack optimizers | EnsAttack, SVRE, AdaEA, SMER with MI-FGSM/PGD |
| Perturbation | L∞ epsilon=8/255, step=2/255, iterations=10 |
| Main comparisons | Five runs; optional refinements disabled |

The six ImageNet surrogates are ResNet-18, Inception-V3, DeiT-Tiny,
ViT-Tiny, EfficientNet-B0, and XCiT-Tiny. Model selection affects
**surrogate composition only**; each ensemble attack retains its own
optimization and gradient-weighting rules.

## ImageNet experiments

Prepare the reference set and select evaluation images using the same
ImageNet class mapping and ordering across checkpoints:

```bash
python experiments/paper_imagenet.py prepare \
  --val-dir dataset/imagenet_validation \
  --test-list results/paper_joint_correct_1000.txt \
  --out-dir results/paper_run

python experiments/paper_imagenet.py scores \
  --val-dir dataset/imagenet_validation \
  --test-list results/paper_joint_correct_1000.txt \
  --out-dir results/paper_run
```

Generate and evaluate transfer attacks:

```bash
python experiments/paper_ensemble.py attack \
  --val-dir dataset/imagenet_validation \
  --test-list results/paper_joint_correct_1000.txt \
  --out-dir results/paper_run \
  --methods ens,svre,adaea,smer \
  --optimization mifgsm \
  --variants fixed,random,C,G,S,C+G,C+S,G+S,C+G+S

python experiments/paper_ensemble.py evaluate \
  --val-dir dataset/imagenet_validation \
  --test-list results/paper_joint_correct_1000.txt \
  --out-dir results/paper_run \
  --methods ens \
  --optimization mifgsm \
  --variants fixed,C+G+S \
  --targets resnet152,vgg19
```

The two targets in this example are for a small evaluation. Use all nine
manuscript target checkpoints for the full experiment. Run different seeds
in independent output directories; record model/checkpoint hashes and exact
input manifests.

The 15 fixed 4-of-6 subsets from the paper are available via
`scripts/run_fixed_sets.py` (dry-run by default).

## Web vision services and multimodal models

AWS Rekognition, Google Cloud Vision, Imagga, Hugging Face inference,
OpenAI, Anthropic, and Gemini evaluations are organized under
`src/desadv/web/` and `notebooks/web/`. Adversarial images are generated
locally, then submitted **only for evaluation** with explicit `--allow-live`.
Results are scored with paired, clean-eligible conditional Top-k ASR.

See [Web evaluation protocol](docs/WEB_PROTOCOL.md) for commands and the
required CSV/JSONL formats.

## CAPTCHA study

The complete **self-hosted CAPTCHA experiment code** is in
[`experiments/captcha/`](experiments/captcha/README.md), including the Gradio
grid/collage challenges, CLIP, OWLv2, YOLOv8, Qwen2.5-VL, LLaVA, and the
local Halligan/GPT-4o runner. It includes the 262-image label manifest,
checkpoint download definitions, per-run logging, and Table 6 CSR reporting.
Dataset images and model weights are distributed separately. See the
[CAPTCHA evaluation protocol](docs/CAPTCHA_PROTOCOL.md) for the setup.

## Reproducibility

See [manuscript-to-code mapping](docs/PAPER_TO_CODE.md),
[experiment requirements](docs/REPRODUCIBILITY.md), and
[data provenance](docs/DATA_PROVENANCE.md).
The published numerical results depend on specific datasets, pretrained
checkpoints, preprocessing, test-set eligibility, random seeds and remote
service versions; these inputs must be recorded to reproduce the tables.

Please observe the venue's double-blind policy before publishing an
identifying GitHub account, domain, or paper link.
