# Table 6 — self-hosted CAPTCHA experiment protocol

**Implementation:** [`experiments/captcha/`](../experiments/captcha/README.md).
The source Gradio app and model runners are now integrated under that directory.
Use only locally hosted challenges under your control.

The manuscript specifies 262 images from ten classes, grid and collage layouts,
CLIP ViT-B/32, OWLv2, YOLOv8, Qwen2.5-VL-3B, LLaVA-1.5-7B and Halligan (GPT-4o).
Each solver–layout–condition uses 500 sessions. Benign and adversarial image
conditions are ε=8/255 and ε=16/255 for the adversarial variants. Source CLI
names `grid` and `area` correspond respectively to grid and collage layouts.

**Required data:** image folders of identical filenames in all conditions,
labels from `experiments/captcha/dataset/captcha_images_labels.csv`, and a
record of how the adversarial conditions were generated and quantized.

Run the local Gradio app with `captcha_gradio_app.py`, then the local browser
solver with `captcha_solver.py` or `halligan/run_gradio.py`. Each runner writes
one `results.csv` per experimental run under `solver_results/`. The paper CSR
summarizer `scripts/captcha_report.py` reads these logs and reports:
`attempted`, `completed`, `errors`, `successes`, and `csr_pct`. Its `--strict`
option requires exactly 500 completed sessions, with no errored attempts.

The `desadv-captcha-evaluate` utility remains available for a normalized JSONL
session manifest with keys `solver`, `layout`, `condition`, `session_id`,
`solved` (boolean).

An evaluation using these scripts alone is **not** sufficient to reproduce
Table 6 without the source images, model weights and raw successful execution
logs. The CAPTCHA study is application-level robustness assessment, and does
not isolate DES-Adv's selection effect or test commercial providers.
