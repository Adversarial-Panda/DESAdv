# Experiment requirements and validation checklist

The codebase implements DES-Adv's selection equations and provides experiment
entrypoints for attack transfer and external-service evaluation. Full table-level
comparisons require the data, weights, response logs, and configuration versions
specified below. Tests of the selection equations do not substitute for a
complete checkpoint-level experiment.

## Manuscript-to-implementation coverage

| Experiment | Code | Inputs / checks |
| --- | --- | --- |
| ImageNet RoC and C/G/S | `experiments/paper_imagenet.py` | 30K DSEL, DINO ViT-B/16, cosine k=7, checkpoint hashes |
| Four-surrogate selection | `src/desadv/selection.py` | Exact normalization and adaptive coefficients |
| EnsAttack / SVRE / AdaEA / SMER | `experiments/ensemble_attacks.py`, `experiments/paper_ensemble.py` | Attack-method parity, preprocessing, hyperparameters |
| Exhaustive fixed sets | `src/desadv/fixed_sets.py` | All 15 subsets from the manuscript |
| CIFAR-10/100 | `experiments/cifar/` | Dataset-specific weights and 14K DSEL manifests required |
| External APIs / VLMs | `src/desadv/web/`, `notebooks/web/` | Endpoint versions, paired responses, label matching |
| Local CAPTCHA | `experiments/captcha/`, `scripts/captcha_report.py` | Original local two-layout app + solver backends + Halligan; 262-image manifest, image tensors, checkpoints and 500 completed sessions per setting |
| Optional failure-case refinements | Appendix-only procedure | Not part of primary experiments |

## Checks before comparing published numerical values

1. Pin pretrained checkpoint names, hashes, framework versions, and exact preprocessing (including Inception handling).
2. Confirm a **jointly benign-correct** set of 1,000 evaluation images for every paper target; verify no DSEL/test overlap.
3. Check image tensor range `[0,1]`, class-index mapping, attack budgets and selected-model order.
4. Validate that SVRE, AdaEA and SMER implementations and their default parameters match the reported attack definitions. In particular, review the SVRE inner-loop behavior and AdaEA's threshold interpretation.
5. Preserve all seed-specific outputs for the five-run comparisons and report the aggregation procedure.
6. Freeze Web-service model IDs, invocation parameters, prompts, label aliases and clean-correct eligibility counts. Never change matching rules based on attack outcomes.
7. Validate any `.pt` to PNG conversion for pixel-range, L∞ perturbation and prediction changes introduced by quantization.
8. For the CAPTCHA table, record each session seed, challenge generator, tile pairing, solver checkpoint, prediction rule and end-to-end completion status.
9. Keep offline eligibility queries and online evaluation strictly separate from attack-generation code.

## Data not included in this repository

Pretrained checkpoint weights, ImageNet/CIFAR datasets, commercial API
credentials, private response logs and original experiment artifacts are not
bundled. When reporting table-level comparisons, reference the exact data and
software versions used. Configuration matching alone does not establish
numerical equivalence.
