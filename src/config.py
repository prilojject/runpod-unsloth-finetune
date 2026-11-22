"""Input schema for the fine-tuning worker.

Every parameter can come from three places, in this order of priority:

1. the job ``input`` object (per request),
2. an environment variable (what the Runpod Hub form sets on the endpoint),
3. the built-in default below.

The module is dependency-free so it can be unit-tested and used for a dry run
without torch, transformers or a GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# HF dataset ids may also be a bare name ("squad") for canonical datasets.
HF_DATASET_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled", ""}


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    default: Any
    env: str | None = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    help: str = ""


PARAMS: list[Param] = [
    # --- model -------------------------------------------------------------
    Param("base_model", str, "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "BASE_MODEL",
          help="Hugging Face model repo id (owner/name)."),
    Param("model_revision", str, "", "MODEL_REVISION", help="Optional model git revision."),
    Param("load_in_4bit", bool, True, "LOAD_IN_4BIT", help="QLoRA (4-bit base). False = 16-bit LoRA."),
    Param("chat_template", str, "auto", "CHAT_TEMPLATE",
          help="'auto' = tokenizer's own template, otherwise an Unsloth template name (chatml, llama-3.1, qwen-2.5, ...)."),
    # --- data --------------------------------------------------------------
    Param("dataset", str, "", "DATASET", help="HF dataset id or http(s) URL to .jsonl/.json/.csv/.parquet."),
    Param("dataset_inline", list, None, None, help="Small list of records sent inline (testing)."),
    Param("dataset_format", str, "auto", "DATASET_FORMAT", choices=("auto", "chat", "alpaca")),
    Param("dataset_split", str, "train", "DATASET_SPLIT"),
    Param("dataset_config", str, "", "DATASET_CONFIG", help="HF dataset config/subset name."),
    Param("max_samples", int, 0, "MAX_SAMPLES", min=0, help="0 = use all rows."),
    Param("skip_invalid_rows", bool, False, "SKIP_INVALID_ROWS",
          help="Drop malformed rows instead of failing the job."),
    Param("system_prompt", str, "", "SYSTEM_PROMPT", help="Optional system prompt prepended to alpaca rows."),
    Param("eval_split", float, 0.0, "EVAL_SPLIT", min=0.0, max=0.5),
    # --- training ----------------------------------------------------------
    Param("num_train_epochs", float, 1.0, "NUM_TRAIN_EPOCHS", min=0.01, max=100),
    Param("max_steps", int, 0, "MAX_STEPS", min=0, help="> 0 overrides epochs (short test runs)."),
    Param("learning_rate", float, 2e-4, "LEARNING_RATE", min=1e-7, max=1e-2),
    Param("per_device_train_batch_size", int, 2, "BATCH_SIZE", min=1, max=256),
    Param("gradient_accumulation_steps", int, 4, "GRAD_ACCUM", min=1, max=1024),
    Param("max_seq_length", int, 2048, "MAX_SEQ_LENGTH", min=128, max=131072),
    Param("warmup_ratio", float, 0.03, "WARMUP_RATIO", min=0.0, max=0.5),
    Param("lr_scheduler_type", str, "cosine", "LR_SCHEDULER",
          choices=("linear", "cosine", "constant", "constant_with_warmup")),
    Param("weight_decay", float, 0.01, "WEIGHT_DECAY", min=0.0, max=1.0),
    Param("packing", bool, False, "PACKING"),
    Param("seed", int, 3407, "SEED", min=0),
    # --- LoRA --------------------------------------------------------------
    Param("lora_r", int, 16, "LORA_R", min=1, max=1024),
    Param("lora_alpha", int, 16, "LORA_ALPHA", min=1, max=2048),
    Param("lora_dropout", float, 0.0, "LORA_DROPOUT", min=0.0, max=0.5),
    Param("target_modules", list, None, "TARGET_MODULES",
          help="List or comma-separated string. Default: all attention + MLP projections."),
    Param("use_rslora", bool, False, "USE_RSLORA"),
    # --- checkpoints / output ---------------------------------------------
    Param("run_name", str, "", "RUN_NAME",
          help="Folder name for checkpoints/adapter. Default: the Runpod job id. Reuse it to resume."),
    Param("output_root", str, "", "OUTPUT_ROOT",
          help="Default: /runpod-volume/unsloth-finetune if a network volume is attached."),
    Param("save_steps", int, 200, "SAVE_STEPS", min=1),
    Param("save_total_limit", int, 2, "SAVE_TOTAL_LIMIT", min=1, max=100),
    Param("resume", bool, True, "RESUME", help="Resume from the last checkpoint of the same run_name."),
    Param("hf_repo_id", str, "", "HF_OUTPUT_REPO", help="Push the adapter here (needs HF_TOKEN)."),
    Param("hf_private", bool, True, "HF_PRIVATE"),
    Param("create_archive", bool, True, "CREATE_ARCHIVE", help="Also write <run_name>-adapter.tar.gz."),
    Param("upload_to_bucket", bool, True, "UPLOAD_TO_BUCKET",
          help="If BUCKET_* env vars are set, upload the archive and return a presigned URL."),
    # --- control -----------------------------------------------------------
    Param("dry_run", bool, False, "DRY_RUN",
          help="Validate input + dataset (sample) and return a report. No model weights, no GPU."),
    Param("dry_run_rows", int, 200, "DRY_RUN_ROWS", min=1, max=100000),
    Param("check_model", bool, True, "CHECK_MODEL",
          help="Dry run: verify the base model exists on HF (metadata only)."),
    Param("logging_steps", int, 10, "LOGGING_STEPS", min=1),
]

PARAM_BY_NAME = {p.name: p for p in PARAMS}

# Aliases accepted in the job input for convenience.
ALIASES = {
    "model": "base_model",
    "model_id": "base_model",
    "format": "dataset_format",
    "epochs": "num_train_epochs",
    "batch_size": "per_device_train_batch_size",
    "grad_accum": "gradient_accumulation_steps",
    "gradient_accumulation": "gradient_accumulation_steps",
    "qlora": "load_in_4bit",
    "output_repo": "hf_repo_id",
    "lr": "learning_rate",
}


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ResolvedConfig:
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def public_dict(self) -> dict[str, Any]:
        """Config safe to log/return (no secrets live here, but drop inline data)."""
        out = dict(self.values)
        if out.get("dataset_inline") is not None:
            out["dataset_inline"] = f"<{len(out['dataset_inline'])} inline rows>"
        return out


def _coerce(param: Param, raw: Any) -> Any:
    if raw is None:
        return None
    t = param.type
    if t is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        raise ValueError(f"{param.name}: expected boolean, got {raw!r}")
    if t is int:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected integer, got boolean")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{param.name}: expected integer, got {raw!r}")
            return int(raw)
        s = str(raw).strip()
        if s == "":
            return param.default
        try:
            return int(float(s)) if re.fullmatch(r"-?\d+(\.0+)?", s) else int(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected integer, got {raw!r}") from exc
    if t is float:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected number, got boolean")
        s = str(raw).strip()
        if s == "":
            return param.default
        try:
            return float(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected number, got {raw!r}") from exc
    if t is list:
        if isinstance(raw, list):
            return raw
        if param.name == "target_modules" and isinstance(raw, str):
            items = [x.strip() for x in raw.split(",") if x.strip()]
            return items or None
        raise ValueError(f"{param.name}: expected a list")
    if t is str:
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected string")
        return str(raw).strip()
    raise TypeError(t)  # pragma: no cover


def resolve_config(job_input: Mapping[str, Any] | None,
                   env: Mapping[str, str] | None = None) -> ResolvedConfig:
    """Merge job input, env vars and defaults; validate; raise ConfigError."""
    env = os.environ if env is None else env
    job_input = dict(job_input or {})
    errors: list[str] = []
    warnings: list[str] = []

    # Normalise aliases.
    for alias, canonical in ALIASES.items():
        if alias in job_input:
            if canonical in job_input:
                errors.append(f"both '{alias}' and '{canonical}' given")
            else:
                job_input[canonical] = job_input.pop(alias)

    unknown = sorted(k for k in job_input if k not in PARAM_BY_NAME)
    if unknown:
        errors.append(f"unknown input field(s): {', '.join(unknown)}")

    for secret_like in ("hf_token", "HF_TOKEN", "token"):
        if secret_like in job_input:
            errors.append("do not send tokens in the job input; set HF_TOKEN as a Runpod secret/env var")

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for p in PARAMS:
        raw, src = None, "default"
        if p.name in job_input and job_input[p.name] is not None:
            raw, src = job_input[p.name], "input"
        elif p.env and env.get(p.env) not in (None, ""):
            raw, src = env.get(p.env), "env"
        try:
            val = _coerce(p, raw) if src != "default" else p.default
        except ValueError as exc:
            errors.append(str(exc))
            val = p.default
        if val is None and p.default is not None:
            val = p.default
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if p.min is not None and val < p.min:
                errors.append(f"{p.name}: must be >= {p.min}, got {val}")
            if p.max is not None and val > p.max:
                errors.append(f"{p.name}: must be <= {p.max}, got {val}")
        if p.choices and val not in p.choices:
            errors.append(f"{p.name}: must be one of {list(p.choices)}, got {val!r}")
        values[p.name] = val
        sources[p.name] = src

    # --- cross-field checks ---------------------------------------------
    if not values["base_model"] or not HF_REPO_RE.match(values["base_model"]):
        errors.append("base_model: expected a Hugging Face repo id like 'owner/name'")

    ds, inline = values["dataset"], values["dataset_inline"]
    if inline is not None:
        if ds and sources["dataset"] == "input":
            errors.append("give either 'dataset' or 'dataset_inline', not both")
        if not isinstance(inline, list) or not inline:
            errors.append("dataset_inline: must be a non-empty list of records")
        elif len(inline) > 5000:
            errors.append("dataset_inline: max 5000 rows; use a URL or HF dataset for more")
        elif not all(isinstance(r, dict) for r in inline):
            errors.append("dataset_inline: every row must be an object")
        values["dataset"] = ""
    elif not ds:
        errors.append("dataset: required (HF dataset id or http(s) URL)")
    elif not (URL_RE.match(ds) or HF_DATASET_RE.match(ds)):
        errors.append("dataset: expected a HF dataset id ('owner/name') or an http(s) URL")

    tm = values["target_modules"]
    if tm is None:
        values["target_modules"] = list(DEFAULT_TARGET_MODULES)
    elif not tm or not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.]+", x) for x in tm):
        errors.append("target_modules: list of module names like ['q_proj','v_proj']")

    if values["hf_repo_id"] and not HF_REPO_RE.match(values["hf_repo_id"]):
        errors.append("hf_repo_id: expected 'owner/name'")

    if values["run_name"] and not RUN_NAME_RE.match(values["run_name"]):
        errors.append("run_name: letters, digits, '.', '_' or '-' (max 100 chars)")

    if values["output_root"] and not values["output_root"].startswith("/"):
        errors.append("output_root: must be an absolute path")

    if values["lora_alpha"] < values["lora_r"] // 4:
        warnings.append("lora_alpha is much smaller than lora_r; updates will be tiny")

    if values["max_steps"] == 0 and values["num_train_epochs"] <= 0:
        errors.append("set num_train_epochs > 0 or max_steps > 0")

    if values["hf_repo_id"] and not env.get("HF_TOKEN"):
        msg = "hf_repo_id is set but HF_TOKEN is not; add HF_TOKEN as a Runpod secret"
        if values["dry_run"]:
            warnings.append(msg)
        else:
            errors.append(msg)

    if errors:
        raise ConfigError(errors)
    return ResolvedConfig(values=values, sources=sources, warnings=warnings)


def effective_batch_size(cfg: ResolvedConfig) -> int:
    return cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps


def estimate_total_steps(cfg: ResolvedConfig, n_train_rows: int) -> int:
    if cfg.max_steps > 0:
        return cfg.max_steps
    per_epoch = max(1, -(-n_train_rows // effective_batch_size(cfg)))  # ceil
    return max(1, int(round(per_epoch * cfg.num_train_epochs)))
