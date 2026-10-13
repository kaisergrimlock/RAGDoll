#!/usr/bin/env python3
"""Run RAGDoll UMBRELA request JSONL through the Jev TypeSafe SDK.

The input is the JSONL produced by trec_dl_csv_to_umbrela.py. One Choice
question is sent per query/passage pair, with labels 0, 1, 2, and 3 matching
the UMBRELA relevance scale. Results are appended after every batch so the
run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import time
from math import ceil
from pathlib import Path


def read_requests(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            query = record.get("query")
            if isinstance(query, dict):
                qid = str(query.get("qid", f"record-{line_number}"))
                query_text = str(query["text"])
            else:
                qid = str(record.get("qid", f"record-{line_number}"))
                query_text = str(query)
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"line {line_number} has no candidate list")
            for candidate_index, candidate in enumerate(candidates):
                doc = candidate.get("doc", candidate) if isinstance(candidate, dict) else {}
                passage = doc.get("segment", candidate.get("text")) if isinstance(doc, dict) else None
                if not isinstance(passage, str):
                    raise ValueError(f"line {line_number}, candidate {candidate_index} has no passage text")
                docid = doc.get("docid", candidate.get("docid")) if isinstance(doc, dict) else None
                task_id = f"{qid}:candidate{candidate_index:04d}"
                rows.append(
                    {
                        "task_id": task_id,
                        "qid": qid,
                        "query": query_text,
                        "docid": None if docid is None else str(docid),
                        "passage": passage,
                    }
                )
    return rows


def choice_value(answer) -> str | None:
    value = getattr(answer, "choice", answer)
    if value is None:
        return None
    value = str(value).strip()
    return value if value in {"0", "1", "2", "3"} else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Limit total query-passage tasks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-probabilities", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    try:
        from typesafe_sdk import Choice, TypeSafeClient
    except ImportError as exc:
        raise SystemExit("Missing TypeSafe SDK. Install it with: pip install typesafe-sdk") from exc
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    if args.overwrite:
        args.output_file.unlink(missing_ok=True)
    tasks = read_requests(args.input_file)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    completed: dict[str, dict] = {}
    if args.output_file.exists():
        with args.output_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed[str(row["task_id"])] = row
    pending = [task for task in tasks if task["task_id"] not in completed]
    print(f"Jev UMBRELA evaluation: {len(tasks)} tasks", flush=True)
    print(f"Already completed: {len(completed)}; pending: {len(pending)}", flush=True)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    client = TypeSafeClient()
    started = time.perf_counter()
    batch_starts = range(0, len(pending), args.batch_size)
    progress = (
        tqdm(batch_starts, total=ceil(len(pending) / args.batch_size) if pending else 0,
             desc="Jev UMBRELA", unit="batch", dynamic_ncols=True)
        if tqdm is not None else batch_starts
    )
    with args.output_file.open("a", encoding="utf-8") as handle:
        failed = 0
        for start in progress:
            batch = pending[start : start + args.batch_size]
            state = {
                "items": [
                    {"id": task["task_id"], "query": task["query"], "passage": task["passage"]}
                    for task in batch
                ]
            }
            questions = {
                task["task_id"]: Choice(
                    instructions=(
                        f"For item {task['task_id']}, assign the UMBRELA relevance grade for the passage "
                        "against its query. Choose exactly one grade based on the definitions."
                    ),
                    criteria={
                        "0": "The passage has nothing to do with the query.",
                        "1": "The passage is related to the query but does not answer it.",
                        "2": "The passage contains an answer or useful information, but it is unclear or mixed with extraneous information.",
                        "3": "The passage is dedicated to the query and contains the exact answer.",
                    },
                )
                for task in batch
            }
            try:
                response = client.system_one(state=state, questions=questions)
                rows = []
                for task in batch:
                    answer = response.answers[task["task_id"]]
                    value = choice_value(answer)
                    row = {
                        **task,
                        "status": "completed" if value is not None else "invalid",
                        "judgment": int(value) if value is not None else -1,
                        "jev_choice": value,
                    }
                    if hasattr(answer, "confidence"):
                        row["confidence"] = answer.confidence
                    if args.include_probabilities and hasattr(answer, "probabilities"):
                        row["probabilities"] = answer.probabilities
                    rows.append(row)
            except Exception as exc:
                failed += 1
                rows = [{**task, "status": "failed", "judgment": -1, "error": str(exc)} for task in batch]
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed[row["task_id"]] = row
            handle.flush()
            if tqdm is not None:
                progress.set_postfix(completed=f"{len(completed)}/{len(tasks)}", failed=failed)
            else:
                print(f"Completed {len(completed)}/{len(tasks)} tasks; failed batches={failed}", flush=True)

    valid = [row for row in completed.values() if row.get("judgment") in {0, 1, 2, 3}]
    summary = {
        "input": str(args.input_file),
        "tasks_requested": len(tasks),
        "completed_or_invalid": len(completed),
        "valid_judgments": len(valid),
        "failed_batches": failed,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "output": str(args.output_file),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
