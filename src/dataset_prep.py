"""Dataset loading, format detection and normalisation (chat / alpaca).

Pure-python helpers (detect/normalise/validate) have no heavy dependencies so
they run in unit tests and in the dry run. Loading uses ``datasets`` lazily.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger(__name__)

CHAT_COLUMNS = ("messages", "conversations", "conversation", "chat")
ROLE_MAP = {
    "system": "system",
    "user": "user", "human": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
    "tool": "tool", "function": "tool", "ipython": "tool",
}
ALPACA_INSTRUCTION_KEYS = ("instruction", "prompt", "question")
ALPACA_OUTPUT_KEYS = ("output", "response", "answer", "completion")
ALPACA_INPUT_KEYS = ("input", "context")

ALPACA_PROMPT = (
    "Below is an instruction that describes a task{ctx_hint}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{input_block}### Response:\n{output}"
)

URL_FILE_TYPES = {
    ".jsonl": "json", ".json": "json", ".ndjson": "json",
    ".csv": "csv", ".parquet": "parquet",
}


class DatasetFormatError(ValueError):
    pass


# --------------------------------------------------------------------------
# detection / normalisation
# --------------------------------------------------------------------------

def _chat_column(record: dict) -> str | None:
    for col in CHAT_COLUMNS:
        if isinstance(record.get(col), list):
            return col
    return None


def _first_key(record: dict, keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in record:
            return k
    return None


def detect_format(record: dict) -> str:
    """Return 'chat' or 'alpaca' for one record, or raise DatasetFormatError."""
    if not isinstance(record, dict):
        raise DatasetFormatError("record is not an object")
    if _chat_column(record):
        return "chat"
    if _first_key(record, ALPACA_INSTRUCTION_KEYS) and _first_key(record, ALPACA_OUTPUT_KEYS):
        return "alpaca"
    raise DatasetFormatError(
        "cannot detect format: expected a 'messages'/'conversations' list (chat) "
        "or 'instruction' + 'output' fields (alpaca); got keys "
        f"{sorted(record.keys())[:10]}"
    )


def detect_dataset_format(records: list[dict]) -> str:
    seen: dict[str, int] = {}
    for r in records[:50]:
        try:
            f = detect_format(r)
        except DatasetFormatError:
            continue
        seen[f] = seen.get(f, 0) + 1
    if not seen:
        detect_format(records[0])  # raises with a helpful message
    return max(seen, key=seen.get)


def normalize_chat(record: dict) -> list[dict[str, str]]:
    col = _chat_column(record)
    if not col:
        raise DatasetFormatError("missing 'messages' (or 'conversations') list")
    turns = record[col]
    if not turns:
        raise DatasetFormatError(f"'{col}' is empty")
    out: list[dict[str, str]] = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            raise DatasetFormatError(f"turn {i} is not an object")
        role = t.get("role", t.get("from"))
        content = t.get("content", t.get("value"))
        if not isinstance(role, str) or role.strip().lower() not in ROLE_MAP:
            raise DatasetFormatError(f"turn {i}: unknown role {role!r}")
        if isinstance(content, list):  # OpenAI-style content parts
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text"]
            content = "".join(texts)
        if not isinstance(content, str) or not content.strip():
            raise DatasetFormatError(f"turn {i}: empty or non-text content")
        out.append({"role": ROLE_MAP[role.strip().lower()], "content": content})
    if not any(t["role"] == "assistant" for t in out):
        raise DatasetFormatError("conversation has no assistant turn to learn from")
    if out[-1]["role"] != "assistant":
        # trailing user turns contribute nothing to SFT; drop them
        while out and out[-1]["role"] != "assistant":
            out.pop()
    return out


def normalize_alpaca(record: dict, system_prompt: str = "") -> list[dict[str, str]]:
    ik = _first_key(record, ALPACA_INSTRUCTION_KEYS)
    ok = _first_key(record, ALPACA_OUTPUT_KEYS)
    if not ik or not ok:
        raise DatasetFormatError("alpaca row needs 'instruction' and 'output'")
    instruction, output = record.get(ik), record.get(ok)
    if not isinstance(instruction, str) or not instruction.strip():
        raise DatasetFormatError("empty 'instruction'")
    if not isinstance(output, str) or not output.strip():
        raise DatasetFormatError("empty 'output'")
    ink = _first_key(record, ALPACA_INPUT_KEYS)
    inp = record.get(ink) if ink else ""
    if inp is not None and not isinstance(inp, str):
        raise DatasetFormatError("'input' must be a string")
    user = instruction.strip() + (f"\n\n{inp.strip()}" if inp and inp.strip() else "")
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
    return msgs


def alpaca_plain_text(record: dict, eos_token: str = "") -> str:
    """Classic Alpaca prompt for base models that have no chat template."""
    ik = _first_key(record, ALPACA_INSTRUCTION_KEYS)
    ok = _first_key(record, ALPACA_OUTPUT_KEYS)
    ink = _first_key(record, ALPACA_INPUT_KEYS)
    inp = (record.get(ink) or "").strip() if ink else ""
    return ALPACA_PROMPT.format(
        ctx_hint=", paired with an input that provides further context" if inp else "",
        instruction=record[ik].strip(),
        input_block=f"### Input:\n{inp}\n\n" if inp else "",
        output=record[ok],
    ) + eos_token


def to_messages(record: dict, fmt: str, system_prompt: str = "") -> list[dict[str, str]]:
    if fmt == "chat":
        msgs = normalize_chat(record)
        if system_prompt and msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        return msgs
    if fmt == "alpaca":
        return normalize_alpaca(record, system_prompt)
    raise DatasetFormatError(f"unsupported format {fmt!r}")


@dataclass
class ValidationReport:
    format: str
    rows_checked: int
    rows_valid: int
    errors: list[str] = field(default_factory=list)  # first N problems
    avg_chars: float = 0.0
    max_chars: int = 0

    @property
    def rows_invalid(self) -> int:
        return self.rows_checked - self.rows_valid

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "rows_checked": self.rows_checked,
            "rows_valid": self.rows_valid,
            "rows_invalid": self.rows_invalid,
            "first_errors": self.errors,
            "avg_chars_per_row": round(self.avg_chars, 1),
            "max_chars_per_row": self.max_chars,
        }


def validate_records(records: list[dict], fmt: str = "auto", system_prompt: str = "",
                     max_errors: int = 10) -> tuple[ValidationReport, list[list[dict]]]:
    """Validate rows; return a report and the normalised message lists (valid rows only)."""
    if not records:
        raise DatasetFormatError("dataset is empty")
    if fmt == "auto":
        fmt = detect_dataset_format(records)
    errors: list[str] = []
    convs: list[list[dict]] = []
    total_chars = 0
    max_chars = 0
    for i, r in enumerate(records):
        try:
            msgs = to_messages(r, fmt, system_prompt)
        except DatasetFormatError as exc:
            if len(errors) < max_errors:
                errors.append(f"row {i}: {exc}")
            continue
        n = sum(len(m["content"]) for m in msgs)
        total_chars += n
        max_chars = max(max_chars, n)
        convs.append(msgs)
    rep = ValidationReport(format=fmt, rows_checked=len(records), rows_valid=len(convs),
                           errors=errors, avg_chars=(total_chars / len(convs)) if convs else 0.0,
                           max_chars=max_chars)
    return rep, convs


def enforce_validation(rep: ValidationReport, skip_invalid_rows: bool) -> None:
    if rep.rows_valid == 0:
        raise DatasetFormatError(
            f"no valid {rep.format} rows; first problems: {rep.errors[:3]}")
    if rep.rows_invalid and not skip_invalid_rows:
        raise DatasetFormatError(
            f"{rep.rows_invalid}/{rep.rows_checked} rows are invalid for format "
            f"'{rep.format}' (set skip_invalid_rows=true to drop them); "
            f"first problems: {rep.errors[:3]}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def url_file_type(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    for ext, kind in URL_FILE_TYPES.items():
        if path.endswith(ext) or path.endswith(ext + ".gz"):
            return kind
    raise DatasetFormatError(
        "dataset URL must end with .jsonl, .json, .csv or .parquet (optionally .gz)")


def download_url(url: str, cache_dir: str, max_bytes: int | None = None) -> str:
    """Download a dataset file once into cache_dir; return the local path."""
    import requests  # lazy

    url_file_type(url)
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(urllib.parse.urlparse(url).path) or "dataset"
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    dest = os.path.join(cache_dir, f"{digest}-{name}")
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and max_bytes is None:
        return dest
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        size = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                size += len(chunk)
                if max_bytes is not None and size >= max_bytes:
                    break
    os.replace(tmp, dest)
    return dest


def load_records(cfg, hf_token: str | None, cache_dir: str, sample_rows: int | None = None) -> list[dict]:
    """Load the dataset as a list of dicts.

    ``sample_rows`` (dry run) streams only the first N rows from HF instead of
    downloading the whole dataset.
    """
    if cfg.dataset_inline is not None:
        rows = list(cfg.dataset_inline)
        return rows[:sample_rows] if sample_rows else rows

    from datasets import load_dataset  # lazy

    ds_id = cfg.dataset
    limit = sample_rows or (cfg.max_samples or None)
    if ds_id.startswith(("http://", "https://")):
        kind = url_file_type(ds_id)
        if sample_rows and kind == "json" and not ds_id.endswith(".gz") and not ds_id.lower().endswith(".json"):
            # JSON Lines: a partial download is enough to validate the head.
            path = download_url(ds_id, os.path.join(cache_dir, "dry-run"), max_bytes=8 << 20)
            rows = []
            import json
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if len(rows) >= sample_rows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # truncated last line of the partial download
            if not rows:
                raise DatasetFormatError("could not parse any JSON Lines rows from the URL")
            return rows
        path = download_url(ds_id, cache_dir)
        ds = load_dataset(kind, data_files=path, split="train", cache_dir=cache_dir)
    else:
        kwargs = dict(split=cfg.dataset_split, token=hf_token or None)
        if cfg.dataset_config:
            kwargs["name"] = cfg.dataset_config
        if sample_rows:
            stream = load_dataset(ds_id, streaming=True, **kwargs)
            return [dict(r) for r, _ in zip(stream, range(sample_rows))]
        ds = load_dataset(ds_id, cache_dir=cache_dir, **kwargs)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]
