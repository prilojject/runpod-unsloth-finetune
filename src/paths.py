"""Where runs, checkpoints, caches and artifacts live."""

from __future__ import annotations

import os
from dataclasses import dataclass

NETWORK_VOLUME = "/runpod-volume"   # Serverless network volume mount
POD_VOLUME = "/workspace"           # Pod volume mount


def default_output_root() -> str:
    if os.path.isdir(NETWORK_VOLUME):
        return os.path.join(NETWORK_VOLUME, "unsloth-finetune")
    # Pods mount their volume at /workspace. Serverless workers (RUNPOD_ENDPOINT_ID set)
    # only persist /runpod-volume, so never treat /workspace as persistent there.
    if (not os.environ.get("RUNPOD_ENDPOINT_ID") and os.path.ismount(POD_VOLUME)
            and os.access(POD_VOLUME, os.W_OK)):
        return os.path.join(POD_VOLUME, "unsloth-finetune")
    return "/tmp/unsloth-finetune"


def is_persistent(path: str) -> bool:
    if path.startswith(NETWORK_VOLUME + "/") and os.path.ismount(NETWORK_VOLUME):
        return True
    return (path.startswith(POD_VOLUME + "/") and not os.environ.get("RUNPOD_ENDPOINT_ID")
            and os.path.ismount(POD_VOLUME))


@dataclass
class RunPaths:
    root: str
    run_dir: str
    checkpoints: str
    adapter: str
    data_cache: str

    @classmethod
    def build(cls, output_root: str, run_name: str) -> "RunPaths":
        root = output_root or default_output_root()
        run_dir = os.path.join(root, "runs", run_name)
        return cls(
            root=root,
            run_dir=run_dir,
            checkpoints=os.path.join(run_dir, "checkpoints"),
            adapter=os.path.join(run_dir, "adapter"),
            data_cache=os.path.join(root, "data-cache"),
        )

    def ensure(self) -> None:
        for p in (self.run_dir, self.checkpoints, self.data_cache):
            os.makedirs(p, exist_ok=True)


def configure_caches() -> None:
    """Keep HF caches on the network volume when present (survives cold starts)."""
    base = NETWORK_VOLUME if os.path.isdir(NETWORK_VOLUME) else None
    if base:
        os.environ.setdefault("HF_HOME", os.path.join(base, "huggingface-cache"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(base, "huggingface-cache", "datasets"))
