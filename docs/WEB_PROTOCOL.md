# Web API, hosted model, and VLM evaluation

- Threat model: **query-free during adversarial generation**; remote services are queried only after final adversarial images exist. This folder imports no gradient/selector models.
- Each clean/attack image has the SAME `image_id`, ground-truth ImageNet label, and method. `predict.py` refuses unpaired manifests.
- Paper: 1000 pairs per service, AWS `DetectLabels(MaxLabels=10,MinConfidence=0)`, Google `LABEL_DETECTION` with 10 labels, Imagga English Top-10 tags, Hugging Face `facebook/convnext-base-224` with `top_k=10`.
- VLMs: five ordered categories as JSON. The paper names GPT-5.6, Claude Sonnet 4.6, Gemini 3.6 Flash. *Availability and exact version/aliases must be verified before runs*. Gemini temperature 0; other exact temperature defaults must be retrieved from historical logs.
- Historical lexical matching: lowercased shared-word or substring. This is permissive, not a fine-grained classification metric. Freeze taxonomy/aliases before evaluating adversarial responses.
- For each `k`, denominator = **number of correctly recognized clean images** (NOT 1000). Missing responses are an error. A service and a rank cutoff have their own eligibility denominator.
- Across methods, strict paired inference requires the **same clean response**, or if repeated, verify hashes and output equivalence. For provider/model/prompt/service drift, report separately.
- Output JSONL includes model string, SHA-256 of source file, timestamp, per-request latency, image_id, method, and split, but never credentials or image bytes.
- Manual/paid call guard: `--allow-live`; `--limit 10` by default; opt in explicitly before 1000-image runs.

Command examples (from repository root):

```bash
python -m pip install -e '.[web,test]'
desadv-web-predict --manifest data/web_manifest.csv --provider aws \
  --output outputs/aws.jsonl --limit 10  # dry run
# Explicit authorization for evaluation queries only:
desadv-web-predict --manifest data/web_manifest.csv --provider aws \
  --output outputs/aws.jsonl --limit 1000 --allow-live

desadv-web-predict --manifest data/web_manifest.csv --provider openai \
  --model gpt-5.6 --output outputs/gpt.jsonl --limit 1000 --allow-live

desadv-web-evaluate --responses outputs/aws.jsonl --ks 1,5,10 \
  --output outputs/aws_summary.csv
```

Model updates, safety filtering, image transformations, output formatting and
sampling can differ from historic runs. Never interpret an SDK smoke test as
replication of paper Table 4 or 5.
