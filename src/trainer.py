"""Unsloth LoRA/QLoRA SFT training. Heavy imports happen inside functions."""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from typing import Any, Callable

from .config import ResolvedConfig, estimate_total_steps
from .dataset_prep import ALPACA_PROMPT
from .paths import RunPaths

log = logging.getLogger(__name__)

# Fields that must match when resuming an existing run directory.
RESUME_KEYS = ("base_model", "load_in_4bit", "lora_r", "lora_alpha", "target_modules",
               "max_seq_length", "dataset", "dataset_format", "use_rslora")


class TrainingError(RuntimeError):
    pass


def check_resume_compat(paths: RunPaths, cfg: ResolvedConfig) -> bool:
    """Return True if there is a compatible checkpoint to resume from."""
    meta_path = os.path.join(paths.run_dir, "run_config.json")
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, encoding="utf-8") as fh:
        prev = json.load(fh)
    diffs = [k for k in RESUME_KEYS if prev.get(k) != cfg.values.get(k)]
    if diffs and cfg.resume:
        raise TrainingError(
            f"run_name '{os.path.basename(paths.run_dir)}' already exists with different "
            f"settings ({', '.join(diffs)}). Use a new run_name or resume=false.")
    return cfg.resume


def write_run_config(paths: RunPaths, cfg: ResolvedConfig) -> None:
    with open(os.path.join(paths.run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg.public_dict(), fh, indent=2, default=str)


def _filter_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs a given TRL/transformers version does not know (API drift guard)."""
    try:
        params = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return kwargs
    dropped = sorted(k for k in kwargs if k not in params)
    if dropped:
        log.info("SFTConfig: ignoring unsupported args for this TRL version: %s", dropped)
    return {k: v for k, v in kwargs.items() if k in params}


def build_texts(convs: list[list[dict]], fmt: str, tokenizer, chat_template: str) -> tuple[list[str], str]:
    """Render conversations to training text. Returns (texts, template_used)."""
    from unsloth.chat_templates import get_chat_template  # lazy

    used = "tokenizer"
    if chat_template and chat_template != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        used = chat_template
    has_template = bool(getattr(tokenizer, "chat_template", None))
    eos = tokenizer.eos_token or ""
    bos = tokenizer.bos_token or ""

    if not has_template and fmt == "alpaca":
        texts = []
        for msgs in convs:
            user = next(m["content"] for m in msgs if m["role"] == "user")
            out = next(m["content"] for m in msgs if m["role"] == "assistant")
            texts.append(ALPACA_PROMPT.format(ctx_hint="", instruction=user, input_block="", output=out) + eos)
        return texts, "alpaca-plain"

    if not has_template:
        log.warning("Model has no chat template; falling back to 'chatml'.")
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        used = "chatml"

    texts = []
    for msgs in convs:
        t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        if bos and t.startswith(bos):
            t = t[len(bos):]  # the tokenizer adds BOS again during tokenisation
        texts.append(t)
    return texts, used


def run_training(cfg: ResolvedConfig, convs: list[list[dict]], fmt: str, paths: RunPaths,
                 hf_token: str | None, progress: Callable[[dict], None]) -> dict[str, Any]:
    import torch  # lazy

    if not torch.cuda.is_available():
        raise TrainingError("CUDA GPU not available; training needs a GPU worker (use dry_run=true to validate only).")

    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402  (must precede trl)
    from datasets import Dataset
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    gpu = torch.cuda.get_device_name(0)
    vram = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    log.info("GPU: %s (%.1f GB)", gpu, vram)
    progress({"stage": "loading_model", "gpu": gpu, "vram_gb": vram})

    t0 = time.time()
    load_kwargs = dict(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
        token=hf_token,
    )
    if cfg.model_revision:
        load_kwargs["revision"] = cfg.model_revision
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=cfg.use_rslora,
    )
    load_s = round(time.time() - t0, 1)

    progress({"stage": "preparing_data"})
    texts, template_used = build_texts(convs, fmt, tokenizer, cfg.chat_template)
    ds = Dataset.from_dict({"text": texts})
    eval_ds = None
    if cfg.eval_split > 0 and len(ds) >= 20:
        split = ds.train_test_split(test_size=cfg.eval_split, seed=cfg.seed)
        ds, eval_ds = split["train"], split["test"]

    total_steps = estimate_total_steps(cfg, len(ds))
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    bf16 = is_bfloat16_supported()
    sft_kwargs = dict(
        output_dir=paths.checkpoints,
        dataset_text_field="text",
        max_length=cfg.max_seq_length,
        max_seq_length=cfg.max_seq_length,  # older TRL name; filtered if unknown
        packing=cfg.packing,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        optim="adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="none",
        dataset_num_proc=2,
    )
    if eval_ds is not None:
        sft_kwargs.update(eval_strategy="steps", eval_steps=cfg.save_steps,
                          per_device_eval_batch_size=cfg.per_device_train_batch_size)
    # max_length and max_seq_length are aliases; keep only the one this TRL knows.
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_length" in sft_params:
        sft_kwargs.pop("max_seq_length", None)
    args = SFTConfig(**_filter_kwargs(SFTConfig, sft_kwargs))

    class ProgressCallback(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if not logs:
                return
            payload = {"stage": "training", "step": state.global_step, "max_steps": state.max_steps,
                       "epoch": round(state.epoch or 0, 3)}
            for k in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                if k in logs:
                    payload[k] = round(float(logs[k]), 6)
            progress(payload)

        def on_save(self, args_, state, control, **kw):
            progress({"stage": "checkpoint_saved", "step": state.global_step})

    trainer_kwargs = dict(model=model, train_dataset=ds, eval_dataset=eval_ds, args=args,
                          callbacks=[ProgressCallback()])
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # pragma: no cover - old TRL
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    resume_from = None
    if cfg.resume:
        last = get_last_checkpoint(paths.checkpoints) if os.path.isdir(paths.checkpoints) else None
        if last:
            resume_from = last
            log.info("Resuming from %s", last)
            progress({"stage": "resuming", "checkpoint": os.path.basename(last)})

    progress({"stage": "training", "step": 0, "max_steps": total_steps})
    t1 = time.time()
    result = trainer.train(resume_from_checkpoint=resume_from)
    train_s = round(time.time() - t1, 1)

    metrics: dict[str, Any] = {k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in (result.metrics or {}).items()}
    if eval_ds is not None:
        try:
            metrics.update({k: round(float(v), 6) for k, v in trainer.evaluate().items()})
        except Exception as exc:  # pragma: no cover
            log.warning("final evaluation failed: %s", exc)

    progress({"stage": "saving_adapter"})
    os.makedirs(paths.adapter, exist_ok=True)
    model.save_pretrained(paths.adapter)
    tokenizer.save_pretrained(paths.adapter)

    return {
        "gpu": gpu,
        "vram_gb": vram,
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "model_load_seconds": load_s,
        "train_seconds": train_s,
        "global_step": int(getattr(result, "global_step", 0)),
        "resumed_from": os.path.basename(resume_from) if resume_from else None,
        "train_rows": len(ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "chat_template": template_used,
        "precision": "bf16" if bf16 else "fp16",
        "metrics": metrics,
    }
