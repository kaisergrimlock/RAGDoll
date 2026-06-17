from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from pi_trec.arena.prompts import render_arena_prompt
from pi_trec.jsonl import read_jsonl


@dataclass(frozen=True)
class AnswerRecord:
    run_id: str
    qid: str
    query: str
    answer_text: str
    source: dict[str, Any]


@dataclass(frozen=True)
class AnswerSet:
    run_id: str
    path: Path
    rows_by_qid: dict[str, AnswerRecord]

    @property
    def qids(self) -> set[str]:
        return set(self.rows_by_qid)


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") if isinstance(row.get("metadata"), dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def qid_of(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(row.get("qid"), row.get("topic_id"), row.get("query_id"), metadata.get("narrative_id"))


def run_id_of(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(row.get("run_id"), metadata.get("run_id"))


def query_of(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(row.get("query"), row.get("topic"), metadata.get("narrative"), metadata.get("title"))


def _answer_sentence_text(answer: Any) -> list[str]:
    if isinstance(answer, str):
        return [answer.strip()] if answer.strip() else []
    if not isinstance(answer, list):
        return []
    texts: list[str] = []
    for item in answer:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = item
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def answer_text_of(row: dict[str, Any]) -> str:
    answer_text = row.get("answer_text")
    if isinstance(answer_text, str) and answer_text.strip():
        return answer_text.strip()
    return " ".join(_answer_sentence_text(row.get("answer"))).strip()


def load_answer_set(path: Path) -> AnswerSet:
    rows_by_qid: dict[str, AnswerRecord] = {}
    run_id: str | None = None
    for row in read_jsonl(path):
        row_run_id = run_id_of(row)
        qid = qid_of(row)
        query = query_of(row)
        answer_text = answer_text_of(row)
        if not row_run_id:
            raise ValueError(f"{path}: missing run_id or metadata.run_id")
        if not qid:
            raise ValueError(f"{path}: missing qid/topic_id/metadata.narrative_id")
        if not query:
            raise ValueError(f"{path}:{qid}: missing query/topic/metadata narrative/title")
        if not answer_text:
            raise ValueError(f"{path}:{qid}: missing answer_text or answer[].text")
        if run_id is None:
            run_id = row_run_id
        elif row_run_id != run_id:
            raise ValueError(f"{path}: inconsistent run_id values: {run_id!r} and {row_run_id!r}")
        if qid in rows_by_qid:
            raise ValueError(f"{path}: duplicate qid {qid!r}")
        rows_by_qid[qid] = AnswerRecord(
            run_id=row_run_id,
            qid=qid,
            query=query,
            answer_text=answer_text,
            source=row,
        )
    if run_id is None:
        raise ValueError(f"{path}: no answer rows found")
    return AnswerSet(run_id=run_id, path=path, rows_by_qid=rows_by_qid)


def load_answer_sets(paths: list[Path]) -> list[AnswerSet]:
    sets = [load_answer_set(path) for path in paths]
    seen: set[str] = set()
    for answer_set in sets:
        if answer_set.run_id in seen:
            raise ValueError(f"duplicate run_id across answer files: {answer_set.run_id!r}")
        seen.add(answer_set.run_id)
    return sets


def assistant_order(seed: int, run_a: str, run_b: str, qid: str) -> tuple[str, str]:
    first, second = sorted((run_a, run_b))
    key = f"{seed}\x00{first}\x00{second}\x00{qid}".encode("utf-8")
    bit = hashlib.sha256(key).digest()[0] % 2
    return (run_a, run_b) if bit == 0 else (run_b, run_a)


def _task_id(run_a: str, run_b: str, qid: str) -> str:
    return f"arena:{run_a}:{run_b}:{qid}"


def iter_arena_tasks(answer_sets: list[AnswerSet], *, seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for left, right in combinations(answer_sets, 2):
        shared_qids = sorted(left.qids & right.qids)
        for qid in shared_qids:
            left_answer = left.rows_by_qid[qid]
            right_answer = right.rows_by_qid[qid]
            if left_answer.query != right_answer.query:
                raise ValueError(
                    f"query mismatch for qid {qid!r} between {left.run_id!r} and {right.run_id!r}"
                )
            assistant_a_run_id, assistant_b_run_id = assistant_order(seed, left.run_id, right.run_id, qid)
            by_run_id = {left.run_id: left_answer, right.run_id: right_answer}
            answer_a = by_run_id[assistant_a_run_id]
            answer_b = by_run_id[assistant_b_run_id]
            tasks.append(
                {
                    "task_id": _task_id(left.run_id, right.run_id, qid),
                    "evaluator": "arena",
                    "system_prompt": "",
                    "instruction": render_arena_prompt(
                        query=left_answer.query,
                        answer_a=answer_a.answer_text,
                        answer_b=answer_b.answer_text,
                    ),
                    "metadata": {
                        "pair": [left.run_id, right.run_id],
                        "qid": qid,
                        "query": left_answer.query,
                        "assistant_a_run_id": assistant_a_run_id,
                        "assistant_b_run_id": assistant_b_run_id,
                    },
                }
            )
    return tasks


def coverage_rows(answer_sets: list[AnswerSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(answer_sets, 2):
        shared = left.qids & right.qids
        rows.append(
            {
                "run_a": left.run_id,
                "run_b": right.run_id,
                "run_a_topics": len(left.qids),
                "run_b_topics": len(right.qids),
                "shared_topics": len(shared),
                "run_a_only_topics": len(left.qids - right.qids),
                "run_b_only_topics": len(right.qids - left.qids),
            }
        )
    return rows
