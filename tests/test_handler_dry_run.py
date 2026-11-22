"""Handler dry runs: no GPU, no model weights, no network (inline data, check_model=false)."""
import json
import os

import pytest

from handler import handler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(*p):
    with open(os.path.join(ROOT, *p)) as fh:
        return json.load(fh)


def test_dry_run_chat_inline():
    out = handler(load_json("test_input.json") | {"id": "t1"})
    assert out["status"] == "ok" and out["dry_run"] is True
    assert out["dataset"]["format"] == "chat" and out["dataset"]["rows_valid"] == 3
    assert out["preview"][-1]["role"] == "assistant"


def test_hub_tests_are_all_dry_runs_and_pass_offline():
    tests = load_json(".runpod", "tests.json")["tests"]
    assert tests, "tests.json must define at least one test"
    for t in tests:
        assert t["input"].get("dry_run") is True, t["name"]
        inp = dict(t["input"], check_model=False)
        if "dataset" in inp:  # network datasets are exercised in the image smoke test
            continue
        out = handler({"id": t["name"], "input": inp})
        assert out.get("status") == "ok", out


def test_invalid_input_returns_error_not_exception():
    out = handler({"id": "t2", "input": {"dataset": "a/b", "lora_r": -1}})
    assert "error" in out and "lora_r" in out["error"]


def test_bad_dataset_returns_error():
    out = handler({"id": "t3", "input": {"dry_run": True, "check_model": False,
                                         "dataset_inline": [{"text": "raw only"}]}})
    assert "error" in out and "cannot detect" in out["error"]


def test_training_without_gpu_fails_cleanly(monkeypatch):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("GPU present; this test must never start training")
    out = handler({"id": "t4", "input": {"check_model": False, "max_steps": 1,
                                         "dataset_inline": [{"instruction": "a", "output": "b"}]}})
    assert "error" in out and "CUDA" in out["error"]


def test_no_secret_in_output(monkeypatch):
    secret = "hf_FAKE_TOKEN_FOR_TEST_1234567890"
    monkeypatch.setenv("HF_TOKEN", secret)
    out = handler({"id": "t5", "input": {"dry_run": True, "check_model": False, "hf_repo_id": "me/x",
                                         "dataset_inline": [{"instruction": "a", "output": "b"}]}})
    assert secret not in json.dumps(out, default=str)
