# AGENTS.md

Instructions for coding agents working in `prilojject/runpod-unsloth-finetune`.

## What this repo is

A Runpod Serverless worker that fine-tunes Hugging Face causal LLMs with Unsloth (LoRA or 4-bit QLoRA, supervised fine-tuning) and returns a LoRA adapter. It is a Hub listing (`type: serverless` in `.runpod/hub.json`), not Unsloth Studio and not a Pod UI.

## Layout

- `handler.py` — Runpod Serverless entry point
- `src/config.py` — job input > env (Hub form) > defaults, validation
- `src/dataset_prep.py` — load and validate chat / alpaca data
- `src/trainer.py` — Unsloth + TRL SFTTrainer, checkpoints, resume
- `src/artifacts.py` — adapter archive, optional S3 upload, optional HF push
- `src/secret_utils.py` — `HF_TOKEN` from env only; redaction in logs
- `.runpod/hub.json`, `.runpod/tests.json` — Hub metadata and official tests
- `examples/` — request bodies and sample datasets
- `scripts/dry_run.py` — local validation; refuses to run unless `"dry_run": true`
- `tests/` — unit tests (no GPU, no weight download)

## Local checks (no GPU, no paid cloud)

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/dry_run.py test_input.json
python handler.py --test_input "$(cat examples/dry_run_alpaca.json)"
```

`scripts/dry_run.py` must never start training. Do not remove the `"dry_run": true` guard. Do not add tests that download model weights or call paid APIs.

## Paid GPU jobs

- Do not start a Runpod pod or Serverless worker unless a human (or an already approved test plan) explicitly asked for a GPU test.
- Keep a GPU test short (smoke config in Hub presets / `examples/smoke_test_request.json`). Delete the endpoint or pod as soon as the test finishes.
- Training jobs must use async `POST /run`, not `/runsync`. Raise `policy.executionTimeout` and `policy.ttl`; the default 600s timeout will kill a real train.
- Attach a network volume if the adapter must survive worker shutdown, or set `hf_repo_id` with `HF_TOKEN` configured as a Runpod secret.

## Secrets

- `HF_TOKEN` and any `BUCKET_*` keys come from environment / Runpod secrets only.
- Never put tokens in job input, git, README examples, logs, or issue comments. The worker rejects tokens in input.
- Do not print `.env`, secret files, or API keys.

## Conventions

- Job input overrides Hub env, which overrides defaults.
- Validate the dataset before any GPU work (already implemented — keep it that way).
- Output is a LoRA adapter only. Do not silently add merge-to-full-weights or GGUF unless a human asked for that feature and tests cover it.
- This worker is text-only SFT, single GPU. DPO / GRPO / vision / multi-GPU are out of scope unless explicitly requested.
- After changing `.runpod/hub.json` or `.runpod/tests.json`, run `pytest` including `tests/test_hub_files.py`.
- Public docs are English. Keep the first paragraph of `README.md` explicit about what the worker is and who it is for. Update `llms.txt` if you add important docs.

## Deploy reminders

- Hub listing URL: `https://console.runpod.io/hub/prilojject/runpod-unsloth-finetune` (live after review).
- A GitHub release is required before a Hub submission.
- Pinned versions live in `requirements.txt`. Do not unpin Unsloth or PyTorch without a rebuild test.
