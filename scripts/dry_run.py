#!/usr/bin/env python3
"""Run the handler locally without the Runpod job loop and print JSON.

Usage:
    python scripts/dry_run.py examples/dry_run_alpaca.json
    python scripts/dry_run.py '{"input": {"dry_run": true, "dataset": "mlabonne/FineTome-100k"}}'

Refuses to run unless the input has dry_run=true (this script never trains).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "test_input.json"
    job = json.load(open(arg)) if os.path.exists(arg) else json.loads(arg)
    job.setdefault("id", "local-dry-run")
    inp = job.setdefault("input", {})
    if str(inp.get("dry_run", "")).lower() not in ("true", "1"):
        print("refusing: this script only runs dry_run=true jobs", file=sys.stderr)
        return 2
    # Keep local scratch files out of the repo / current machine's volumes.
    os.environ.setdefault("OUTPUT_ROOT", "/tmp/unsloth-dry-run")
    from handler import handler
    out = handler(job)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 1 if "error" in out else 0


if __name__ == "__main__":
    sys.exit(main())
