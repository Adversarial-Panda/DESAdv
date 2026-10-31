# Manuscript-to-code index

| Manuscript component | Source | Checks / caveats |
| --- | --- | --- |
| Fig. 1: DSEL -> embeddings -> FAISS | `experiments/paper_imagenet.py::prepare` | Verify disjoint DSEL/test content |
| Eq. (1) RoC cosine Top-k | `get_neighbors` in `paper_imagenet.py` | k=7, normalized DINO B/16 |
| Eq. (2) competence C | `compute_scores` | Unweighted fraction correct across RoC |
| Eq. (3) pseudo-label gradient | `pseudo_label_input_grad` | Per-surrogate argmax at CLEAN input |
| Eq. (4) diversity G | `compute_scores` | Includes diagonal self cosine |
| Eq. (5-7) gradient stability S | `compute_scores` | SAME pseudo-label used for noisy gradient |
| Eq. (8) signal normalization | `criterion_matrix`, `src/desadv/selection.py` | `(C, G/2, (1+S)/2)` |
| Eq. (9) coefficients | `adaptive_signal_coefficients`, `src/desadv/selection.py` | a0=(.4,.5,.1), lambda=.5 |
| Eq. (10) subset of four | `select_models` | Selection once before attack |
| Eq. (11) attack | `experiments/paper_ensemble.py` | Native attack weighting |
| Table 8 fixed sets | `src/desadv/fixed_sets.py` | All 15 exact combinations |
| Tables 4, 5 Web/VLM | `src/desadv/web/` | Conditional Top-k, no live calls in CI |
| Table 6 custom CAPTCHA | `experiments/captcha/`, `scripts/captcha_report.py`, `src/desadv/captcha/evaluate.py` | Source grid and collage app, 262-image manifest, CLIP/OWLv2/YOLO/VLM/Halligan local solvers; 500 completed sessions/condition plus weights/images required |
| Appendix D refinements | none | Off in all primary comparisons |

Model weights, data manifests, API response JSONL, and raw evaluation logs must be versioned for table-level comparisons.
