import csv
import json
import sys
from pathlib import Path

import pytest

from pi_trec.arena import (
    fit_arena_ratings,
    iter_arena_tasks,
    leaderboard_rows,
    load_answer_set,
    load_answer_sets,
    pairwise_rows,
    parse_verdict,
)
from pi_trec.arena.stages import assistant_order, coverage_rows
from pi_trec.cli import main


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_load_answer_set_accepts_normalized_rows(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [{"run_id": "sys-a", "qid": "q1", "query": "question", "answer_text": "answer"}],
    )

    answer_set = load_answer_set(path)

    assert answer_set.run_id == "sys-a"
    assert answer_set.rows_by_qid["q1"].answer_text == "answer"


def test_load_answer_set_accepts_official_trec_rows_and_concatenates_sentences(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {
                "metadata": {"run_id": "sys-a", "narrative_id": "q1", "narrative": "question"},
                "answer": [
                    {"text": "Sentence one.", "citations": [0]},
                    {"text": "Sentence two.", "citations": [1]},
                ],
                "references": ["DO_NOT_RENDER"],
            }
        ],
    )

    answer_set = load_answer_set(path)

    assert answer_set.rows_by_qid["q1"].answer_text == "Sentence one. Sentence two."


def test_arena_prompt_uses_answer_text_without_citations_or_references(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "a.jsonl",
        [
            {
                "metadata": {"run_id": "sys-a", "narrative_id": "q1", "narrative": "question"},
                "answer": [{"text": "Left sentence.", "citations": ["DO_NOT_RENDER_CITATION"]}],
                "references": ["DO_NOT_RENDER_REFERENCE"],
            }
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [
            {
                "metadata": {"run_id": "sys-b", "narrative_id": "q1", "narrative": "question"},
                "answer": [{"text": "Right sentence.", "citations": ["DO_NOT_RENDER_CITATION"]}],
                "references": ["DO_NOT_RENDER_REFERENCE"],
            }
        ],
    )

    task = iter_arena_tasks(load_answer_sets([left, right]), seed=13)[0]

    assert "Left sentence." in task["instruction"]
    assert "Right sentence." in task["instruction"]
    assert "DO_NOT_RENDER_CITATION" not in task["instruction"]
    assert "DO_NOT_RENDER_REFERENCE" not in task["instruction"]


def test_pairing_uses_shared_qids_and_reports_coverage(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "one", "answer_text": "a1"},
            {"run_id": "sys-a", "qid": "q2", "query": "two", "answer_text": "a2"},
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [{"run_id": "sys-b", "qid": "q1", "query": "one", "answer_text": "b1"}],
    )
    answer_sets = load_answer_sets([left, right])

    tasks = iter_arena_tasks(answer_sets, seed=13)
    coverage = coverage_rows(answer_sets)

    assert [task["metadata"]["qid"] for task in tasks] == ["q1"]
    assert coverage[0]["shared_topics"] == 1
    assert coverage[0]["run_a_only_topics"] == 1
    assert coverage[0]["run_b_only_topics"] == 0


def test_load_answer_set_rejects_duplicate_qids(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"},
            {"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "b"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate qid"):
        load_answer_set(path)


def test_load_answer_set_rejects_inconsistent_run_ids(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"},
            {"run_id": "sys-b", "qid": "q2", "query": "q", "answer_text": "b"},
        ],
    )

    with pytest.raises(ValueError, match="inconsistent run_id"):
        load_answer_set(path)


def test_iter_arena_tasks_rejects_query_mismatch(tmp_path: Path) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "left", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "right", "answer_text": "b"}])

    with pytest.raises(ValueError, match="query mismatch"):
        iter_arena_tasks(load_answer_sets([left, right]), seed=13)


def test_assistant_order_is_deterministic_and_recorded(tmp_path: Path) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])

    task = iter_arena_tasks(load_answer_sets([left, right]), seed=13)[0]

    assert assistant_order(13, "sys-a", "sys-b", "q1") == (
        task["metadata"]["assistant_a_run_id"],
        task["metadata"]["assistant_b_run_id"],
    )


def test_parse_verdict_requires_single_label() -> None:
    assert parse_verdict("[[A]]") == "A"
    assert parse_verdict(" [[B]]\n") == "B"
    assert parse_verdict("[[Tie]]") == "Tie"
    assert parse_verdict("I choose [[A]]") is None
    assert parse_verdict("[[C]]") is None


def test_pairwise_and_arena_ranking() -> None:
    judgments = [
        {"status": "completed", "pair": ["a", "b"], "judge_verdict": "A", "preferred_run_id": "a"},
        {"status": "completed", "pair": ["a", "b"], "judge_verdict": "Tie", "preferred_run_id": None},
        {"status": "completed", "pair": ["a", "c"], "judge_verdict": "A", "preferred_run_id": "a"},
        {"status": "completed", "pair": ["b", "c"], "judge_verdict": "A", "preferred_run_id": "b"},
    ]
    coverage = [
        {"run_a": "a", "run_b": "b", "shared_topics": 2},
        {"run_a": "a", "run_b": "c", "shared_topics": 1},
        {"run_a": "b", "run_b": "c", "shared_topics": 1},
    ]

    pairwise = pairwise_rows(judgments, coverage)
    scores = fit_arena_ratings(judgments, ["a", "b", "c"])
    leaderboard = leaderboard_rows(judgments, ["a", "b", "c"])

    assert pairwise[0]["run_a_preference_rate"] == 0.75
    assert scores["a"] > scores["b"] > scores["c"]
    assert round(sum(scores.values()) / len(scores), 6) == 1000.0
    assert [row["run_id"] for row in leaderboard] == ["a", "b", "c"]
    assert "arena_score" in leaderboard[0]


def test_materialize_arena_cli(tmp_path: Path, monkeypatch) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "materialize",
            "arena",
            "--answers",
            str(left),
            "--answers",
            str(right),
            "--output-file",
            str(output),
        ],
    )

    main()

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["evaluator"] == "arena"
    assert row["metadata"]["pair"] == ["sys-a", "sys-b"]


def test_materialize_arena_cli_accepts_answers_dir(tmp_path: Path, monkeypatch) -> None:
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    _write(answers_dir / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])
    _write(answers_dir / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "materialize",
            "arena",
            "--answers-dir",
            str(answers_dir),
            "--output-file",
            str(output),
        ],
    )

    main()

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["metadata"]["pair"] == ["sys-a", "sys-b"]


def test_arena_compare_all_cli_with_fake_pi(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"[[A]]"}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    left = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q1", "answer_text": "a1"},
            {"run_id": "sys-a", "qid": "q2", "query": "q2", "answer_text": "a2"},
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [
            {"run_id": "sys-b", "qid": "q1", "query": "q1", "answer_text": "b1"},
            {"run_id": "sys-b", "qid": "q2", "query": "q2", "answer_text": "b2"},
        ],
    )
    output_dir = tmp_path / "arena"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "arena",
            "compare-all",
            "--answers",
            str(left),
            "--answers",
            str(right),
            "--output-dir",
            str(output_dir),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--model",
            "judge-model",
            "--thinking",
            "medium",
            "--max-concurrency",
            "1",
            "--no-cache",
            "--overwrite",
        ],
    )

    main()

    judgments = [json.loads(line) for line in (output_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    leaderboard = list(csv.DictReader((output_dir / "leaderboard.csv").open(encoding="utf-8")))
    assert len(judgments) == 2
    assert all(row["judge_verdict"] == "A" for row in judgments)
    assert all(row["model"] == "judge-model" for row in judgments)
    assert len(leaderboard) == 2
    assert (output_dir / "tasks.jsonl").exists()
    assert (output_dir / "pairwise.csv").exists()
    assert (output_dir / "coverage.csv").exists()


def test_arena_compare_all_cli_accepts_answers_dir(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"[[Tie]]"}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    _write(answers_dir / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q1", "answer_text": "a1"}])
    _write(answers_dir / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q1", "answer_text": "b1"}])
    output_dir = tmp_path / "arena"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "arena",
            "compare-all",
            "--answers-dir",
            str(answers_dir),
            "--output-dir",
            str(output_dir),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--max-concurrency",
            "1",
            "--no-cache",
            "--overwrite",
        ],
    )

    main()

    row = json.loads((output_dir / "judgments.jsonl").read_text(encoding="utf-8"))
    assert row["judge_verdict"] == "Tie"
