"""Runpod Serverless entry point: Unsloth LoRA/QLoRA fine-tuning.

Local dry run (no GPU, no model download):
    python handler.py --test_input '{"input": {"dry_run": true, "dataset_inline": [...]}}'
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import ConfigError, estimate_total_steps, effective_batch_size, resolve_config  # noqa: E402
from src.dataset_prep import DatasetFormatError, enforce_validation, load_records, validate_records  # noqa: E402
from src.paths import RunPaths, configure_caches, is_persistent  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
log = logging.getLogger("unsloth-worker")


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod
                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # progress is best effort
                log.debug("progress_update failed: %s", exc)
    return send


def _preview(msgs: list[dict], limit: int = 400) -> list[dict]:
    return [{"role": m["role"], "content": m["content"][:limit] + ("..." if len(m["content"]) > limit else "")}
            for m in msgs]


def _check_model(base_model: str, revision: str, token: str | None) -> dict[str, Any]:
    """Metadata-only check that the model exists and is reachable (no weights)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=token).model_info(base_model, revision=revision or None)
        return {"exists": True, "gated": bool(getattr(info, "gated", False)),
                "private": bool(getattr(info, "private", False))}
    except Exception as exc:
        msg = redact(str(exc)).splitlines()[0][:300]
        if "401" in msg or "404" in msg:
            msg = "model not found, private or gated (gated/private models need HF_TOKEN with access): " + msg
        return {"exists": False, "error": msg}


def _check_hf_write(token: str | None) -> dict[str, Any]:
    if not token:
        return {"token_configured": False}
    try:
        from huggingface_hub import HfApi
        who = HfApi(token=token).whoami()
        return {"token_configured": True, "user": who.get("name")}
    except Exception as exc:
        return {"token_configured": True, "error": redact(str(exc)).splitlines()[0][:300]}


def handler(job: dict) -> dict[str, Any]:
    started = time.time()
    job_input = job.get("input") or {}
    try:
        cfg = resolve_config(job_input)
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    paths = RunPaths.build(cfg.output_root, run_name)
    hf_token = get_hf_token()
    progress = _progress_fn(job)

    # --- dataset: load + validate before any GPU work ----------------------
    try:
        progress({"stage": "validating_dataset"})
        records = load_records(cfg, hf_token, paths.data_cache,
                               sample_rows=cfg.dry_run_rows if cfg.dry_run else None)
        report, convs = validate_records(records, cfg.dataset_format, cfg.system_prompt)
        enforce_validation(report, cfg.skip_invalid_rows)
    except DatasetFormatError as exc:
        return {"error": f"dataset: {exc}"}
    except Exception as exc:
        log.error("dataset load failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset load failed: {redact(str(exc)).splitlines()[0][:500]}"}

    common = {
        "worker_version": __version__,
        "run_name": run_name,
        "config": cfg.public_dict(),
        "config_sources": {k: v for k, v in cfg.sources.items() if v != "default"},
        "warnings": list(cfg.warnings),
        "dataset": report.as_dict(),
        "output_root": paths.root,
        "persistent_storage": is_persistent(paths.root),
    }
    if not common["persistent_storage"]:
        common["warnings"].append(
            "no network volume detected: checkpoints/adapter live on the worker disk and are lost "
            "when the worker stops; attach a network volume or set hf_repo_id / bucket upload")

    if cfg.dry_run:
        n_rows = report.rows_valid
        common.update({
            "dry_run": True,
            "status": "ok",
            "preview": _preview(convs[0]),
            "effective_batch_size": effective_batch_size(cfg),
            "estimated_steps_for_sampled_rows": estimate_total_steps(cfg, n_rows),
            "note": ("dataset counts are for the sampled rows only" if cfg.dataset_inline is None else
                     "inline dataset fully checked"),
            "hf_output": _check_hf_write(hf_token) if cfg.hf_repo_id else None,
            "bucket_upload_configured": all(os.environ.get(k) for k in (
                "BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")),
            "elapsed_seconds": round(time.time() - started, 2),
        })
        if cfg.check_model:
            common["base_model_check"] = _check_model(cfg.base_model, cfg.model_revision, hf_token)
            if not common["base_model_check"]["exists"]:
                common["status"] = "error"
                return {"error": f"base_model not reachable: {common['base_model_check'].get('error')}",
                        **{k: common[k] for k in ("dataset", "config")}}
        return common

    # --- real training --------------------------------------------------------
    from src.artifacts import publish
    from src.trainer import TrainingError, check_resume_compat, run_training, write_run_config

    try:
        paths.ensure()
        check_resume_compat(paths, cfg)
        write_run_config(paths, cfg)
        train_info = run_training(cfg, convs, report.format, paths, hf_token, progress)
        progress({"stage": "publishing"})
        artifacts = publish(cfg, paths, run_name, hf_token)
    except TrainingError as exc:
        return {"error": str(exc), "run_name": run_name}
    except Exception as exc:
        log.error("training failed: %s", redact(traceback.format_exc()))
        return {"error": f"training failed: {redact(str(exc)).splitlines()[0][:500]}",
                "run_name": run_name,
                "checkpoints_dir": paths.checkpoints,
                "hint": "resubmit with the same run_name to resume from the last checkpoint"}

    common.update({
        "status": "completed",
        "training": train_info,
        "artifacts": artifacts,
        "elapsed_seconds": round(time.time() - started, 1),
        # free GPU memory fully before the next job
        "refresh_worker": True,
    })
    return common


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
