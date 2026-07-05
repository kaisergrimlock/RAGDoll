import json
from pathlib import Path

import pytest

from ragdoll.config import RunConfig, resolve_thinking
from ragdoll.validate import ValidateConfig, check_row, detect_kind, validate


def test_detect_kind() -> None:
    assert detect_kind({"query": "q", "candidates": ["a"]}) == "create"
    assert detect_kind({"query": "q", "nuggets": [{"text": "n"}]}) == "nuggets"
    assert detect_kind({"nuggets": [{"text": "n", "assignment": "support"}]}) == "assignment"
    assert detect_kind({"answer_record": {}, "nugget_record": {}}) == "assign"
    assert detect_kind({"qid": "q1", "answer_text": "x"}) == "answers"


def test_check_row_flags_missing_assignment() -> None:
    errors = check_row("assignment", {"nuggets": [{"text": "n"}]})
    assert any("assignment" in e for e in errors)
    assert check_row("assignment", {"nuggets": [{"text": "n", "assignment": "support"}]}) == []


def test_validate_passes_clean_file(tmp_path: Path, capsys) -> None:
    path = tmp_path / "nuggets.jsonl"
    path.write_text(json.dumps({"query": "q", "nuggets": [{"text": "n"}]}) + "\n", encoding="utf-8")
    validate(ValidateConfig(input_file=path))
    assert "is valid" in capsys.readouterr().out


def test_validate_raises_on_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")  # create kind w/o candidates
    with pytest.raises(SystemExit):
        validate(ValidateConfig(input_file=path, kind="create"))


def test_resolve_thinking_profile() -> None:
    assert resolve_thinking("openai-codex/gpt-5.5", "auto") == "minimal"
    assert resolve_thinking("openai-codex/gpt-5.4-mini", "auto") == "minimal"
    assert resolve_thinking("openai-codex/gpt-4o", "auto") == "medium"
    assert resolve_thinking("openai-codex/gpt-5.5", "high") == "high"  # explicit wins


def test_run_config_defaults_to_medium_for_gpt5() -> None:
    cfg = RunConfig(model="openai-codex/gpt-5.4-mini", temperature=0.2)
    agent = cfg.local_agent_config()
    assert agent.thinking == "medium"
    assert agent.temperature == 0.2


def test_run_config_auto_resolves_minimal_for_gpt5() -> None:
    cfg = RunConfig(model="openai-codex/gpt-5.4-mini", thinking="auto")
    assert cfg.local_agent_config().thinking == "minimal"
