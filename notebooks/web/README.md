# Web vision evaluation notebooks

These notebooks show service-specific data collection for AWS Rekognition,
Google Cloud Vision, Imagga, Hugging Face, GPT, Claude and Gemini. The reusable
collector and conditional Top-k ASR implementation live in `src/desadv/web/`.

Before running a notebook, configure the required environment variable for
its provider. The notebooks contain no execution outputs or saved credentials.
Run only against inputs and endpoints for which you have authorization; check
API availability, model versions, label output formats and billing limits.

Use paired clean/adversarial images with identical IDs and fixed matching
rules. See `docs/WEB_PROTOCOL.md` for the evaluation protocol.
