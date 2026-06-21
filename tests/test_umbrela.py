import asyncio
import json
from pathlib import Path

from pi_trec.config import UmbrelaJudgeConfig
from pi_trec.umbrela import iter_prompt_tasks, judge, parse_umbrela_judgment, render_umbrela_prompt


def _fake_agent(tmp_path: Path, output_text: str) -> Path:
    agent = tmp_path / "fake_pi.py"
    payload = json.dumps({"type": "message_end", "message": {"role": "assistant", "content": output_text}})
    agent.write_text(f"#!/usr/bin/env python3\nprint({json.dumps(payload)})\n", encoding="utf-8")
    agent.chmod(0o755)
    return agent


def _rows_by_task_id(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["task_id"], []).append(row)
    return grouped


def _source_prompt(prompt_type: str) -> str:
    source = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "upstream"
        / "umbrela"
        / f"qrel_zeroshot_{prompt_type}.yaml"
    )
    text = source.read_text(encoding="utf-8")
    expected = text.split("prefix_user: |-\n", 1)[1]
    return "\n".join(line[2:] if line.startswith("  ") else line for line in expected.splitlines())


def test_umbrela_bing_prompt_matches_source_template() -> None:
    expected = _source_prompt("bing")
    assert render_umbrela_prompt(query="q", passage="p", prompt_type="bing") == expected.format(query="q", passage="p")


def test_umbrela_basic_prompt_matches_source_template() -> None:
    expected = _source_prompt("basic")
    assert render_umbrela_prompt(query="q", passage="p", prompt_type="basic") == expected.format(query="q", passage="p")


def test_umbrela_materializes_shared_query_candidate_schema() -> None:
    tasks = iter_prompt_tasks(
        [
            {
                "query": {"qid": "q1", "text": "what is python"},
                "candidates": [
                    "Python is a language.",
                    {"doc": {"docid": "d2", "segment": "Coffee is a drink."}},
                ],
            }
        ],
        prompt_type="basic",
    )
    assert len(tasks) == 2
    assert tasks[0]["metadata"]["qid"] == "q1"
    assert tasks[1]["metadata"]["docid"] == "d2"
    assert "Query: what is python" in tasks[0]["instruction"]


def test_parse_umbrela_judgment() -> None:
    assert parse_umbrela_judgment("##final score: 3") == 3
    assert parse_umbrela_judgment("The relevance category is 2") == 2
    assert parse_umbrela_judgment("nope") is None


def test_materialized_task_is_json_serializable() -> None:
    task = iter_prompt_tasks([{"query": "q", "candidates": ["p"]}], prompt_type="bing")[0]
    json.dumps(task)


def test_umbrela_judge_resume_skips_completed(tmp_path: Path) -> None:
    records = [{"query": {"qid": "q1", "text": "what is python"}, "candidates": ["Python is a language.", "Coffee is a drink."]}]
    input_path = tmp_path / "in.jsonl"
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    tasks = iter_prompt_tasks(records, prompt_type="bing")
    assert len(tasks) == 2
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps(
            {"task_id": tasks[0]["task_id"], "status": "completed", "query": "what is python", "passage": "Python is a language.", "judgment": 3}
        )
        + "\n",
        encoding="utf-8",
    )
    config = UmbrelaJudgeConfig(
        input_file=input_path,
        output_file=output_path,
        resume=True,
        cache_dir=None,
        agent_binary=str(_fake_agent(tmp_path, "##final score: 2")),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
    )
    asyncio.run(judge(config))
    grouped = _rows_by_task_id(output_path)
    assert grouped[tasks[0]["task_id"]] == [
        {"task_id": tasks[0]["task_id"], "status": "completed", "query": "what is python", "passage": "Python is a language.", "judgment": 3}
    ]
    assert len(grouped[tasks[1]["task_id"]]) == 1
    assert grouped[tasks[1]["task_id"]][0]["judgment"] == 2
