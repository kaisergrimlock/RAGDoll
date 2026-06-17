import asyncio
import json
from pathlib import Path

from pi_trec.config import SupportJudgeConfig
from pi_trec.support import (
    SUPPORT_EVAL_PROMPT,
    iter_support_tasks,
    judge,
    parse_support_label,
    render_support_prompt,
)


def _fake_agent(tmp_path: Path, output_text: str) -> Path:
    agent = tmp_path / "fake_pi.py"
    payload = json.dumps({"type": "message_end", "message": {"role": "assistant", "content": output_text}})
    agent.write_text(f"#!/usr/bin/env python3\nprint({json.dumps(payload)})\n", encoding="utf-8")
    agent.chmod(0o755)
    return agent


def test_support_prompt_matches_source_script() -> None:
    source = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "upstream"
        / "trec2024-rag"
        / "support_evaluation_individual_gpt4o.py"
    )
    text = source.read_text(encoding="utf-8")
    expected = text.split('SUPPORT_EVAL_PROMPT = """', 1)[1].split('"""', 1)[0]
    assert SUPPORT_EVAL_PROMPT == expected
    assert render_support_prompt(statement="s", citation="c") == expected.format(statement="s", citation="c")


def test_support_tasks_accept_direct_rows() -> None:
    tasks = iter_support_tasks([{"task_id": "t", "statement": "s", "citation": "c"}])
    assert tasks[0]["task_id"] == "t"
    assert "Statement: s" in tasks[0]["instruction"]


def test_support_tasks_accept_resolved_trec_answer_rows() -> None:
    tasks = iter_support_tasks(
        [
            {
                "topic_id": "q1",
                "run_id": "r1",
                "references": ["d1"],
                "segments": {"d1": "citation text"},
                "answer": [{"text": "statement", "citations": [0]}],
            }
        ]
    )
    assert tasks[0]["task_id"] == "r1:q1:s0:c0"
    assert tasks[0]["metadata"]["citation"] == "citation text"


def test_parse_support_label() -> None:
    assert parse_support_label("Full Support") == "FS"
    assert parse_support_label("partial support") == "PS"
    assert parse_support_label("No Support") == "NS"
    assert parse_support_label("unknown") is None


def test_support_judge_resume_skips_completed(tmp_path: Path) -> None:
    records = [
        {"task_id": "a", "statement": "s1", "citation": "c1"},
        {"task_id": "b", "statement": "s2", "citation": "c2"},
    ]
    input_path = tmp_path / "in.jsonl"
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"task_id": "a", "status": "completed", "statement": "s1", "citation": "c1", "support_label": "NS"}) + "\n",
        encoding="utf-8",
    )
    config = SupportJudgeConfig(
        input_file=input_path,
        output_file=output_path,
        resume=True,
        cache_dir=None,
        agent_binary=str(_fake_agent(tmp_path, "Full Support")),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
    )
    asyncio.run(judge(config))
    grouped: dict[str, list[dict]] = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["task_id"], []).append(row)
    assert grouped["a"] == [{"task_id": "a", "status": "completed", "statement": "s1", "citation": "c1", "support_label": "NS"}]
    assert len(grouped["b"]) == 1
    assert grouped["b"][0]["support_label"] == "FS"

