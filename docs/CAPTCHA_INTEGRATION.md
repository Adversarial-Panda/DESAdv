# Custom CAPTCHA evaluation: code-to-manuscript index

This self-hosted experiment is reported in Section 5.3 and Table 6 of the
ACM Web 2027 manuscript. Run only with your own local Gradio application.

| Paper component | Source file / configuration |
| --- | --- |
| 262 images across 10 classes | `experiments/captcha/dataset/captcha_images_labels.csv` |
| 3×3 grid challenge | `experiments/captcha/captcha_gradio_app.py --type grid` |
| Collage-based object localization | `experiments/captcha/area_captcha.py`, app `--type area` |
| Benign, ε=8/255, ε=16/255 | App `--images benign`, `adv`, `adv2` (image content must be verified) |
| CLIP ViT-B/32 | `experiments/captcha/solver_backends.py`, `--model clip` |
| OWLv2 | `--model owlv2` (collage) |
| YOLOv8 | `--model yolov8`, `--model yolov8-cam` (where applicable) |
| Qwen2.5-VL-3B | `--model qwen2.5-vl` or `qwen2.5-vl-ground` |
| LLaVA-1.5-7B | `--model llava` or `llava-ground` |
| Halligan (GPT-4o) | `experiments/captcha/halligan/run_gradio.py`, Halligan source and MIT license |
| CSR (%), N = 500 | `scripts/captcha_report.py --expected 500 --strict` |
| More detailed per-target/tile metrics | `experiments/captcha/compare_results.py` |
| Image pairing sanity checks | `scripts/check_captcha_data.py --require-all` |

## Experiments and interpretation

The manuscripts' CAPTCHA study measures the **impact of adversarial images on
custom CAPTCHA solvers**, not the isolated contribution of DES-Adv selection
nor attacks against deployed third-party CAPTCHA services. For each layout,
solver and condition, preserve raw `results.csv` from 500 complete sessions,
pretrained checkpoint revisions, randomized challenge settings, and
corresponding image manifest (the same image IDs across conditions).

The code ZIP ships the 262 image labels and the evaluation implementation;
the actual images, the adversarial image-generation artifacts, the pretrained
model weights and the paper's 500-session measurement logs were not part of
the supplied archive. Numerical results must be generated from these inputs.

The Halligan source files needed for the local Gradio runner are included with
the original MIT license, without unrelated third-party deployed-provider
challenge examples, cloned Git histories, or unused DINOv2 development trees.
Upstream CLIP source required by Halligan's pinned environment is included
with its own license. The main research environment and Halligan's pinned
Python 3.12 environment are intentionally separate.

## Source integrity and safety

The supplied local solver enforces localhost URLs. Do not modify that
restriction to run against access-control systems you do not own or have
explicit permission to assess. No API key, service session token, private
image, or pretrained weight is needed to browse this source repository.
