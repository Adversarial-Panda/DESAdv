# Image and model provenance

No proprietary datasets, pretrained checkpoint weights, API keys or private
provider responses are part of this repository.

## Required paths (user supplies)

```
dataset/imagenet_validation/<1000 WordNet class directories>/*.JPEG
results/paper_joint_correct_1000.txt    # absolute/relative image paths, exact order
results/eligibility.csv                 # sample_id,target,clean_correct (optional audit)
results/paper_run/                      # created by experiments
```

The 30,000 DSEL samples must be held out from the 1,000 selected test
samples; use the original global class-to-index mapping. A file named
`hard_cases_missed_by_mobilenet.txt` is not automatically an all-target-correct
paper evaluation set.

## Adversarial .pt conversion

Original tensor outputs may use `0.pt` through `999.pt` in selected-image
order with CHW floating RGB, values [0,1]. `scripts/convert_pt_to_png.py`
converts those to **lossless PNG** and produces a manifest with EMPTY labels.
Fill labels from the verified ImageNet taxonomy rather than assuming WNID
folder names are suitable English labels. 8-bit PNG quantization can change
the exact pixel perturbation and *must be rechecked*.

Also provide SHA-256 manifest of weights, a requirements lock, and a map from
each result table to its raw CSV/JSONL inputs before release.
