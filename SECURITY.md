# Responsible research scope

This repository concerns query-free adversarial-transfer **research and evaluation**.
The attack-generation scripts operate on local, user-controlled surrogate models.
No production service is queried during selection or adversarial optimization.
Cloud/VLM adapters are separated under `src/desadv/web/` and demand explicit
`--allow-live` authorization for any network request; users must comply with
provider terms, billing, consent, and applicable policies.

The CAPTCHA scripts are limited to private/authorized custom challenges and do
not implement interactions with production reCAPTCHA/hCaptcha installations.
Never include API credentials, private response logs, third-party datasets or
provider secrets in a public repository. Scan git history before release.
