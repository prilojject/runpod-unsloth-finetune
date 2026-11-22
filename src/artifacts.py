"""Publish the trained adapter: tar.gz archive, optional S3 presigned URL, optional HF push."""

from __future__ import annotations

import logging
import os
import tarfile
from typing import Any

from .secret_utils import redact

log = logging.getLogger(__name__)

BUCKET_ENV = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")


def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV)


def make_archive(adapter_dir: str, dest_path: str) -> str:
    tmp = dest_path + ".part"
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(adapter_dir, arcname="adapter")
    os.replace(tmp, dest_path)
    return dest_path


def upload_archive_to_bucket(archive_path: str, run_name: str) -> str:
    """Upload via the Runpod SDK helper (boto3); returns a presigned URL."""
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket  # lazy

    bucket = os.environ.get("BUCKET_NAME") or None
    return upload_file_to_bucket(
        file_name=os.path.basename(archive_path),
        file_location=archive_path,
        bucket_name=bucket,
        prefix=f"unsloth-finetune/{run_name}",
    )


def push_adapter_to_hf(adapter_dir: str, repo_id: str, private: bool, token: str,
                       commit_message: str) -> str:
    from huggingface_hub import HfApi  # lazy

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=adapter_dir, repo_id=repo_id, repo_type="model",
                      commit_message=commit_message)
    return f"https://huggingface.co/{repo_id}"


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / 2**20, 2)


def publish(cfg, paths, run_name: str, hf_token: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "adapter_dir": paths.adapter,
        "adapter_size_mb": dir_size_mb(paths.adapter),
        "adapter_files": sorted(os.listdir(paths.adapter)),
    }
    if cfg.create_archive or (cfg.upload_to_bucket and bucket_configured()):
        archive = os.path.join(paths.run_dir, f"{run_name}-adapter.tar.gz")
        out["archive_path"] = make_archive(paths.adapter, archive)
        if cfg.upload_to_bucket and bucket_configured():
            try:
                out["download_url"] = upload_archive_to_bucket(archive, run_name)
            except Exception as exc:
                out["bucket_error"] = redact(str(exc))[:500]
                log.error("bucket upload failed: %s", redact(str(exc)))
    if cfg.hf_repo_id:
        if not hf_token:
            out["hf_error"] = "HF_TOKEN not configured; adapter kept on disk only"
        else:
            try:
                out["hf_repo_url"] = push_adapter_to_hf(
                    paths.adapter, cfg.hf_repo_id, cfg.hf_private, hf_token,
                    commit_message=f"LoRA adapter for {cfg.base_model} (run {run_name})")
            except Exception as exc:
                out["hf_error"] = redact(str(exc))[:500]
                log.error("HF push failed: %s", redact(str(exc)))
    return out
