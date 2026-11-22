"""Static checks of .runpod/hub.json and tests.json against the publishing guide."""
import json
import os

from src.config import PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ENV = {"HF_TOKEN", "RUNPOD_INIT_TIMEOUT"}  # consumed outside the config schema


def load(name):
    with open(os.path.join(ROOT, ".runpod", name)) as fh:
        return json.load(fh)


def test_required_repo_files_exist():
    for f in ("handler.py", "Dockerfile", "README.md", "LICENSE", ".runpod/hub.json", ".runpod/tests.json"):
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_hub_json_required_fields():
    hub = load("hub.json")
    for k in ("title", "description", "type", "category", "config"):
        assert hub.get(k), k
    assert hub["type"] == "serverless"
    assert hub["category"] in {"audio", "embedding", "language", "video", "image"}
    cfg = hub["config"]
    assert cfg["runsOn"] == "GPU"
    assert isinstance(cfg["containerDiskInGb"], int)
    assert isinstance(cfg["gpuCount"], int) and cfg["gpuIds"]


def test_hub_env_keys_are_consumed_by_worker():
    known = {p.env for p in PARAMS if p.env} | EXTRA_ENV
    keys = [e["key"] for e in load("hub.json")["config"]["env"]]
    assert len(keys) == len(set(keys))
    assert set(keys) <= known, set(keys) - known
    for preset in load("hub.json")["config"]["presets"]:
        assert set(preset["defaults"]) <= set(keys), preset["name"]


def test_hub_inputs_well_formed():
    for e in load("hub.json")["config"]["env"]:
        inp = e["input"]
        assert inp["type"] in {"string", "number", "boolean", "huggingface"}
        assert inp.get("name") and inp.get("description")
        if inp["type"] == "number":
            assert inp["min"] <= inp["default"] <= inp["max"]
        if inp["type"] == "boolean":
            assert {inp["trueValue"], inp["falseValue"]} == {"true", "false"}


def test_tests_json_is_safe():
    t = load("tests.json")
    assert t["config"]["gpuTypeId"] and t["config"]["gpuCount"] == 1
    env = {e["key"]: e["value"] for e in t["config"].get("env", [])}
    assert env.get("DRY_RUN") == "true"  # belt and braces: Hub tests never train
    for case in t["tests"]:
        assert case["name"] and case["input"]["dry_run"] is True
        assert case.get("timeout", 0) >= 60000
