import json
import os

import pytest

from src.dataset_prep import (DatasetFormatError, alpaca_plain_text, detect_dataset_format,
                              detect_format, enforce_validation, normalize_alpaca, normalize_chat,
                              url_file_type, validate_records)

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def load(name):
    with open(os.path.join(EX, name)) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_detect():
    assert detect_format({"messages": [{"role": "user", "content": "x"}]}) == "chat"
    assert detect_format({"conversations": []}) == "chat"
    assert detect_format({"instruction": "a", "output": "b"}) == "alpaca"
    assert detect_format({"prompt": "a", "response": "b"}) == "alpaca"
    with pytest.raises(DatasetFormatError, match="cannot detect"):
        detect_format({"text": "raw"})


def test_detect_dataset_uses_majority():
    rows = [{"instruction": "a", "output": "b"}] * 3 + [{"messages": []}]
    assert detect_dataset_format(rows) == "alpaca"


def test_normalize_chat_openai_and_sharegpt():
    msgs = normalize_chat({"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Yo"}]})
    assert msgs == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]
    parts = normalize_chat({"messages": [{"role": "user", "content": [{"type": "text", "text": "A"}]},
                                         {"role": "assistant", "content": "B"}]})
    assert parts[0]["content"] == "A"


def test_normalize_chat_drops_trailing_user_turn():
    msgs = normalize_chat({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                                        {"role": "user", "content": "unanswered"}]})
    assert msgs[-1]["role"] == "assistant" and len(msgs) == 2


@pytest.mark.parametrize("row,needle", [
    ({"messages": []}, "empty"),
    ({"messages": [{"role": "user", "content": "only user"}]}, "no assistant"),
    ({"messages": [{"role": "wizard", "content": "x"}]}, "unknown role"),
    ({"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "a"}]}, "empty"),
    ({"messages": ["not a dict"]}, "not an object"),
])
def test_bad_chat_rows(row, needle):
    with pytest.raises(DatasetFormatError, match=needle):
        normalize_chat(row)


def test_normalize_alpaca_with_and_without_input():
    m = normalize_alpaca({"instruction": "Translate", "input": "Hello", "output": "Bonjour"})
    assert m == [{"role": "user", "content": "Translate\n\nHello"}, {"role": "assistant", "content": "Bonjour"}]
    m = normalize_alpaca({"instruction": "Say hi", "input": "", "output": "hi"}, system_prompt="Be nice")
    assert m[0] == {"role": "system", "content": "Be nice"} and m[1]["content"] == "Say hi"


@pytest.mark.parametrize("row", [
    {"instruction": "", "output": "x"},
    {"instruction": "x", "output": "  "},
    {"instruction": "x"},
    {"instruction": "x", "output": "y", "input": 5},
])
def test_bad_alpaca_rows(row):
    with pytest.raises(DatasetFormatError):
        normalize_alpaca(row)


def test_alpaca_plain_text():
    t = alpaca_plain_text({"instruction": "Add", "input": "2+2", "output": "4"}, eos_token="</s>")
    assert "### Instruction:\nAdd" in t and "### Input:\n2+2" in t and t.endswith("### Response:\n4</s>")
    assert "### Input" not in alpaca_plain_text({"instruction": "Hi", "output": "Yo"})


def test_example_files_validate():
    rep, convs = validate_records(load("sample_chat.jsonl"))
    assert rep.format == "chat" and rep.rows_valid == 3 and len(convs) == 3
    rep, convs = validate_records(load("sample_alpaca.jsonl"), "alpaca")
    assert rep.rows_valid == 3 and convs[0][1]["role"] == "assistant"


def test_wrong_format_is_rejected_before_training():
    rep, _ = validate_records(load("sample_chat.jsonl"), "alpaca")
    with pytest.raises(DatasetFormatError, match="no valid alpaca rows"):
        enforce_validation(rep, skip_invalid_rows=False)


def test_partial_invalid_strict_vs_skip():
    rows = [{"instruction": "a", "output": "b"}, {"instruction": "", "output": "b"}]
    rep, convs = validate_records(rows, "alpaca")
    assert rep.rows_invalid == 1 and len(convs) == 1
    with pytest.raises(DatasetFormatError, match="skip_invalid_rows"):
        enforce_validation(rep, skip_invalid_rows=False)
    enforce_validation(rep, skip_invalid_rows=True)


def test_empty_dataset():
    with pytest.raises(DatasetFormatError, match="empty"):
        validate_records([])


def test_url_file_type():
    assert url_file_type("https://x.com/a/train.jsonl?download=1") == "json"
    assert url_file_type("https://x.com/a.parquet") == "parquet"
    assert url_file_type("https://x.com/a.csv.gz") == "csv"
    with pytest.raises(DatasetFormatError):
        url_file_type("https://x.com/a.txt")
