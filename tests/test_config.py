import pytest

from src.config import (DEFAULT_TARGET_MODULES, ConfigError, estimate_total_steps,
                        resolve_config)

INLINE = [{"instruction": "hi", "output": "hello"}]


def cfg(inp, env=None):
    return resolve_config(inp, env or {})


def test_defaults_for_7b_qlora():
    c = cfg({"dataset": "mlabonne/FineTome-100k"})
    assert c.base_model == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    assert c.load_in_4bit is True
    assert (c.lora_r, c.lora_alpha, c.max_seq_length) == (16, 16, 2048)
    assert (c.per_device_train_batch_size, c.gradient_accumulation_steps) == (2, 4)
    assert c.target_modules == DEFAULT_TARGET_MODULES
    assert c.dry_run is False


def test_env_used_and_input_overrides_env():
    env = {"BASE_MODEL": "unsloth/Llama-3.2-3B-Instruct", "LORA_R": "32", "LOAD_IN_4BIT": "false",
           "DATASET": "owner/ds", "MAX_STEPS": "50"}
    c = cfg({"lora_r": 8}, env)
    assert c.base_model == "unsloth/Llama-3.2-3B-Instruct"
    assert c.lora_r == 8 and c.sources["lora_r"] == "input"
    assert c.load_in_4bit is False and c.sources["load_in_4bit"] == "env"
    assert c.max_steps == 50


def test_empty_env_values_fall_back_to_default():
    c = cfg({"dataset": "a/b"}, {"LORA_R": "", "HF_OUTPUT_REPO": ""})
    assert c.lora_r == 16 and c.hf_repo_id == ""


def test_aliases():
    c = cfg({"model": "a/b", "dataset": "c/d", "epochs": 2, "qlora": False, "format": "chat"})
    assert c.base_model == "a/b" and c.num_train_epochs == 2.0
    assert c.load_in_4bit is False and c.dataset_format == "chat"


@pytest.mark.parametrize("inp,needle", [
    ({"dataset": "a/b", "foo": 1}, "unknown input field"),
    ({"dataset": "a/b", "lora_r": 0}, "lora_r"),
    ({"dataset": "a/b", "lora_r": "x"}, "lora_r"),
    ({"dataset": "a/b", "dataset_format": "sharegpt"}, "dataset_format"),
    ({"dataset": "a/b", "base_model": "llama"}, "base_model"),
    ({"dataset": "not a dataset!"}, "dataset"),
    ({}, "dataset: required"),
    ({"dataset": "a/b", "dataset_inline": INLINE}, "not both"),
    ({"dataset_inline": []}, "non-empty"),
    ({"dataset": "a/b", "hf_repo_id": "noslash"}, "hf_repo_id"),
    ({"dataset": "a/b", "run_name": "../etc"}, "run_name"),
    ({"dataset": "a/b", "output_root": "relative/path"}, "absolute"),
    ({"dataset": "a/b", "eval_split": 0.9}, "eval_split"),
    ({"dataset": "a/b", "hf_token": "secret"}, "tokens"),
    ({"dataset": "a/b", "load_in_4bit": "maybe"}, "boolean"),
    ({"dataset": "a/b", "max_steps": 1.5}, "integer"),
])
def test_invalid_inputs(inp, needle):
    with pytest.raises(ConfigError) as exc:
        cfg(inp)
    assert needle in str(exc.value)


def test_push_without_token_fails_for_training_but_warns_in_dry_run():
    with pytest.raises(ConfigError, match="HF_TOKEN"):
        cfg({"dataset": "a/b", "hf_repo_id": "me/x"})
    c = cfg({"dataset": "a/b", "hf_repo_id": "me/x", "dry_run": True})
    assert any("HF_TOKEN" in w for w in c.warnings)
    assert cfg({"dataset": "a/b", "hf_repo_id": "me/x"}, {"HF_TOKEN": "x" * 20}).hf_repo_id == "me/x"


def test_url_dataset_and_bool_strings():
    c = cfg({"dataset": "https://example.com/d.jsonl", "packing": "yes", "resume": "0"})
    assert c.packing is True and c.resume is False


def test_target_modules_from_comma_string():
    c = cfg({"dataset": "a/b"}, {"TARGET_MODULES": "q_proj, v_proj"})
    assert c.target_modules == ["q_proj", "v_proj"]


def test_inline_dataset_clears_env_dataset_and_public_dict_hides_rows():
    c = cfg({"dataset_inline": INLINE}, {"DATASET": "owner/ds"})
    assert c.dataset == ""
    assert c.public_dict()["dataset_inline"] == "<1 inline rows>"


def test_estimate_steps():
    c = cfg({"dataset": "a/b", "num_train_epochs": 2})
    assert estimate_total_steps(c, 80) == 20  # 80 rows / (2*4) * 2 epochs
    c = cfg({"dataset": "a/b", "max_steps": 30})
    assert estimate_total_steps(c, 10_000) == 30
