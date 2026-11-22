# Unsloth LoRA / QLoRA Fine-Tuning — Runpod Serverless worker

Fine-tune an open LLM from the Hugging Face Hub on your own data with
[Unsloth](https://github.com/unslothai/unsloth) (LoRA or 4-bit QLoRA, supervised fine-tuning),
on a Runpod Serverless GPU. You send a base model and a dataset; you get back a **LoRA adapter**,
saved to your network volume and (optionally) pushed to the Hugging Face Hub.

Source: `https://github.com/prilojject/runpod-unsloth-finetune` (MIT).

---

## What you send / what you get

**You send** (job `input`, or defaults from the Hub deploy form):

| Field | Example | Notes |
|---|---|---|
| `base_model` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Any HF causal LM supported by Unsloth (Llama, Qwen, Mistral, Gemma, Phi, ...). |
| `dataset` | `mlabonne/FineTome-100k` or `https://.../train.jsonl` | HF dataset id **or** an http(s) URL to `.jsonl/.json/.csv/.parquet` (optionally `.gz`). |
| `dataset_format` | `chat` / `alpaca` / `auto` | See formats below. |
| `hf_repo_id` | `your-name/my-adapter` | Optional. Pushes the adapter at the end of a **successful** run. Needs `HF_TOKEN`. |

**You get** (job `output`):

```json
{
  "status": "completed",
  "run_name": "my-run",
  "training": {"global_step": 625, "train_seconds": 5400, "peak_vram_gb": 14.2, "metrics": {"train_loss": 0.91}},
  "artifacts": {
    "adapter_dir": "/runpod-volume/unsloth-finetune/runs/my-run/adapter",
    "archive_path": "/runpod-volume/unsloth-finetune/runs/my-run/my-run-adapter.tar.gz",
    "hf_repo_url": "https://huggingface.co/your-name/my-adapter",
    "download_url": "https://... (only if BUCKET_* is configured)"
  }
}
```

The adapter folder contains `adapter_model.safetensors`, `adapter_config.json` and the tokenizer files —
load it with PEFT/Unsloth on top of the same base model, or merge it.

## Dataset formats

**chat** — one conversation per row, OpenAI or ShareGPT style (`messages`, `conversations`, `conversation`):

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

**alpaca** — `instruction`, optional `input`, `output` (also accepts `prompt`/`response`):

```json
{"instruction": "Translate to French.", "input": "Good morning", "output": "Bonjour"}
```

Rows are rendered with the model's own chat template (`chat_template: "auto"`), or with an Unsloth template
you name (`chatml`, `llama-3.1`, `qwen-2.5`, ...). Alpaca data on a base model without a chat template uses the
classic Alpaca prompt. **Every row is validated before any GPU work starts**; malformed rows fail the job
immediately with the first problems listed (or are dropped with `skip_invalid_rows: true`).

## How to run a job (important: long jobs)

Fine-tuning takes minutes to hours, so always use the **async** `/run` route and raise the timeout:

* Runpod's default **execution timeout is 600 s** — set it higher on the endpoint (Edit endpoint → Advanced → Execution timeout)
  or per request with `policy.executionTimeout` (max 7 days).
* Job **TTL defaults to 24 h** and covers queue + run time — raise `policy.ttl` for longer runs (max 7 days).
* Async results are kept for 30 min after completion — the adapter itself is on your volume / HF, so nothing is lost.

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "input": {
      "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
      "dataset": "mlabonne/FineTome-100k",
      "dataset_format": "chat",
      "max_samples": 5000,
      "num_train_epochs": 1,
      "run_name": "qwen7b-finetome-5k",
      "hf_repo_id": "your-name/qwen7b-finetome-lora"
    },
    "policy": {"executionTimeout": 43200000, "ttl": 86400000}
  }'
# then poll: GET https://api.runpod.ai/v2/$ENDPOINT_ID/status/<job id>  (progress shows step/loss)
```

**Validate first, for free-ish:** add `"dry_run": true`. The worker checks your parameters, streams the first
`dry_run_rows` rows of the dataset, validates the format, checks the model exists on HF (metadata only) and
returns a report with a preview — no model weights are downloaded and no training runs.

## Parameters

Job input overrides the Hub form (environment variables), which override the defaults.

| Input | Env (Hub form) | Default | Meaning |
|---|---|---|---|
| `base_model` | `BASE_MODEL` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | HF model id |
| `load_in_4bit` | `LOAD_IN_4BIT` | `true` | QLoRA; `false` = 16-bit LoRA |
| `dataset` / `dataset_inline` | `DATASET` | — | HF id or URL / small inline list (≤5000 rows) |
| `dataset_format` | `DATASET_FORMAT` | `auto` | `chat`, `alpaca`, `auto` |
| `dataset_split`, `dataset_config` | `DATASET_SPLIT`, `DATASET_CONFIG` | `train`, — | HF split / subset |
| `max_samples` | `MAX_SAMPLES` | `0` (all) | Limit rows |
| `num_train_epochs` | `NUM_TRAIN_EPOCHS` | `1` | Epochs |
| `max_steps` | `MAX_STEPS` | `0` | >0 overrides epochs (quick tests) |
| `learning_rate` | `LEARNING_RATE` | `2e-4` | |
| `per_device_train_batch_size` | `BATCH_SIZE` | `2` | |
| `gradient_accumulation_steps` | `GRAD_ACCUM` | `4` | effective batch = 8 |
| `max_seq_length` | `MAX_SEQ_LENGTH` | `2048` | longer rows are truncated |
| `lora_r`, `lora_alpha`, `lora_dropout` | `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT` | `16`, `16`, `0` | |
| `target_modules` | `TARGET_MODULES` | q,k,v,o,gate,up,down proj | list or comma string |
| `eval_split` | `EVAL_SPLIT` | `0` | hold-out fraction for eval loss |
| `packing`, `use_rslora`, `warmup_ratio`, `lr_scheduler_type`, `weight_decay`, `seed` | same in caps | | standard SFT knobs |
| `run_name` | `RUN_NAME` | job id | folder for checkpoints/adapter; reuse to resume |
| `save_steps`, `save_total_limit` | `SAVE_STEPS`, `SAVE_TOTAL_LIMIT` | `200`, `2` | checkpointing |
| `resume` | `RESUME` | `true` | continue from the last checkpoint of `run_name` |
| `hf_repo_id`, `hf_private` | `HF_OUTPUT_REPO`, `HF_PRIVATE` | —, `true` | push adapter at the end |
| `create_archive`, `upload_to_bucket` | `CREATE_ARCHIVE`, `UPLOAD_TO_BUCKET` | `true`, `true` | tar.gz + optional S3 presigned URL |
| `dry_run`, `dry_run_rows`, `check_model` | `DRY_RUN`, ... | `false`, `200`, `true` | validation only |

## Storage, checkpoints and resume

Attach a **network volume** to the endpoint (it mounts at `/runpod-volume`). The worker then keeps the HF model
cache, checkpoints (every `save_steps`) and the final adapter under `/runpod-volume/unsloth-finetune/runs/<run_name>/`.
If a job is interrupted (worker preempted, timeout), resubmit with the **same `run_name`** and it resumes from the
last checkpoint. Without a volume everything lives on the worker disk and is lost when the worker stops — then set
`hf_repo_id` (or a bucket) so the adapter leaves the worker.

To download files from a network volume without starting a GPU, use Runpod's
[S3-compatible API](https://docs.runpod.io/storage/s3-api) or `runpodctl`. Alternatively set `BUCKET_ENDPOINT_URL`,
`BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY` (and optional `BUCKET_NAME`) to have the archive uploaded to your
S3-compatible bucket and a presigned `download_url` returned.

## Secrets

* `HF_TOKEN` — needed for gated/private base models and for pushing. Create a Runpod secret (e.g. `hf_token`) and
  set the endpoint env `HF_TOKEN={{ RUNPOD_SECRET_hf_token }}`. Never send tokens in the job input — the worker rejects them.
* Tokens are never logged or returned; log lines are filtered for secret values.

## Recommended GPUs

| Model size | Mode | VRAM | GPUs |
|---|---|---|---|
| ≤3B | 16-bit LoRA or QLoRA | 12–24 GB | RTX 4090 / L4 / A5000 (24 GB) |
| 7–8B | QLoRA (default) | ~10–18 GB at 2k ctx | RTX 4090, L4, A5000, L40/L40S |
| 7–8B | 16-bit LoRA | ~24–40 GB | L40S / A6000 (48 GB) |
| 13–14B | QLoRA | ~16–30 GB | L40S / A6000 (48 GB) |
| 30–34B | QLoRA | ~30–45 GB | A100 / H100 80 GB |
| 70B | QLoRA | ~45–75 GB | A100 / H100 80 GB, H200 |

Longer `max_seq_length` and larger batch sizes need more memory. The default Hub configuration allows 24 GB and
48 GB pools; pick 80 GB GPUs for 30B+ models in the endpoint settings.

## Why queue-based Serverless (not a load balancer endpoint)

* Queue-based endpoints support execution timeouts up to 7 days with async `/run` jobs and retries.
* Load-balancing endpoints cap each HTTP request at ~5.5 minutes — too short for training.
* The Hub currently lists Serverless repos; this worker also runs on a Pod with the same image if you prefer
  (`python handler.py --rp_serve_api` exposes the same API locally).

## Limitations

* Text-only SFT (no vision models, no DPO/GRPO) and single-GPU training.
* Output is a LoRA adapter only (no merged weights / GGUF export in this version).
* Loss is computed on the full rendered conversation (prompt + response).
* One training job per worker at a time; set max workers to the number of parallel runs you want.

## Local development (no GPU)

```bash
pip install -r requirements-dev.txt
pytest -q                                            # unit tests: schema + dataset prep + dry runs
python scripts/dry_run.py test_input.json            # dry run via the handler, prints JSON
python handler.py --test_input "$(cat examples/dry_run_alpaca.json)"   # same through the Runpod SDK
docker build -t runpod-unsloth-finetune:dev .
```

`scripts/dry_run.py` refuses inputs without `"dry_run": true`, so it can never start training.

## License

MIT — see [LICENSE](LICENSE). Unsloth, TRL, Transformers and the models you train have their own licenses.
