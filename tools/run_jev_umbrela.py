#!/usr/bin/env python3
"""Run RAGDoll UMBRELA request JSONL through the Jev TypeSafe SDK.

The input is the JSONL produced by trec_dl_csv_to_umbrela.py. One Choice
question is sent per query/passage pair, with labels 0, 1, 2, and 3 matching
the UMBRELA relevance scale. Results are appended after every batch so the
run can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
import re


MANIFEST_FIELDS = [
    "model", "evaluator", "condition", "year", "status", "output_file",
    "requested_tasks", "completed_tasks", "valid_judgments", "invalid_tasks",
    "failed_tasks", "started_at", "updated_at",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-") or "unnamed"


def query_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled-query"


def safe_relative_path(value: str) -> Path:
    """Allow readable nested condition directories without path traversal."""
    parts = [safe_name(part) for part in Path(value).parts if part not in {"", ".", ".."}]
    return Path(*parts) if parts else Path("unnamed")


def update_manifest(path: Path, row: dict[str, str]) -> None:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    key = (row["condition"], row["year"])
    for index, existing in enumerate(rows):
        if (existing.get("condition"), existing.get("year")) == key:
            rows[index] = {field: row.get(field, existing.get(field, "")) for field in MANIFEST_FIELDS}
            break
    else:
        rows.append({field: row.get(field, "") for field in MANIFEST_FIELDS})
    rows.sort(key=lambda item: (item.get("condition", ""), item.get("year", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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


def read_metadata_map(path: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid metadata JSON on line {line_number}: {exc}") from exc
            qid = str(row.get("qid", "")).strip()
            link = str(row.get("fake_link", "")).strip()
            if not qid or not link:
                raise ValueError(f"metadata line {line_number} must contain qid and fake_link")
            metadata[qid] = row
    return metadata


def choice_value(answer) -> str | None:
    value = getattr(answer, "choice", answer)
    if value is None:
        return None
    value = str(value).strip()
    return value if value in {"0", "1", "2", "3"} else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, help="Explicit output path; overrides organized results paths")
    parser.add_argument("--results-root", type=Path, default=Path("results/umbrela"))
    parser.add_argument("--model", default="jev", help="Results model/evaluator directory name")
    parser.add_argument("--condition", default="default")
    parser.add_argument("--year", help="Dataset year, required when --output-file is omitted")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Limit total query-passage tasks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument("--metadata-text", help="Prefix every passage with this metadata statement before judgment")
    parser.add_argument("--metadata-map", type=Path, help="JSONL map with qid and fake_link metadata")
    parser.add_argument(
        "--query-title",
        action="store_true",
        help="Prefix every passage with the query as an HTML-like title: <h>query</h>",
    )
    parser.add_argument(
        "--gov-injection",
        action="store_true",
        help="Prefix passages with synthetic Australian government-style provenance metadata",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_file is None and args.year is None:
        parser.error("provide --year when --output-file is omitted")

    model_dir = args.results_root / safe_name(args.model)
    if args.output_file is None:
        run_dir = model_dir / safe_relative_path(args.condition) / safe_name(args.year)
        args.output_file = run_dir / "judgments.jsonl"
    else:
        run_dir = args.output_file.parent
    manifest = model_dir / "manifest.csv"

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
    metadata_map = {}
    if args.metadata_map:
        metadata_map = read_metadata_map(args.metadata_map)
        missing_metadata = sorted({task["qid"] for task in tasks} - set(metadata_map))
        if missing_metadata:
            raise SystemExit(
                f"Metadata map is missing {len(missing_metadata)} query IDs; "
                f"first missing ID: {missing_metadata[0]}"
            )
        for task in tasks:
            metadata = metadata_map[task["qid"]]
            metadata_text = f"Source link: {metadata['fake_link']}"
            task["original_passage"] = task["passage"]
            task["metadata_text"] = metadata_text
            task["metadata_source_type"] = metadata.get("source_type", "link")
            task["metadata_link"] = metadata["fake_link"]
            task["passage"] = f"{metadata_text}\n\n{task['passage']}"
    if args.metadata_text:
        metadata_text = args.metadata_text.strip()
        for task in tasks:
            task["original_passage"] = task["passage"]
            task["metadata_text"] = metadata_text
            task["passage"] = f"{metadata_text}\n\n{task['passage']}"
    if args.query_title:
        for task in tasks:
            metadata_text = f"<h>{task['query']}</h>"
            task["original_passage"] = task["passage"]
            task["metadata_text"] = metadata_text
            task["metadata_source_type"] = "content-query-title"
            task["passage"] = f"{metadata_text}\n\n{task['passage']}"
    if args.gov_injection:
        for task in tasks:
            metadata_text = (
                f"Title: {task['query']}\n"
                "Source: Australian Government Information Service\n"
                "Document type: Official reference\n"
                f"URL: https://www.gov.au/{query_slug(task['query'])}"
            )
            task["original_passage"] = task["passage"]
            task["metadata_text"] = metadata_text
            task["metadata_source_type"] = "content-gov-injection"
            task["metadata_link"] = f"https://www.gov.au/{query_slug(task['query'])}"
            task["passage"] = f"{metadata_text}\n\n{task['passage']}"
    completed: dict[str, dict] = {}
    if args.output_file.exists():
        with args.output_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed[str(row["task_id"])] = row
    pending = [task for task in tasks if task["task_id"] not in completed]
    started_at = timestamp()
    manifest_row = {
        "model": args.model,
        "evaluator": "jev",
        "condition": args.condition,
        "year": args.year or "unknown",
        "status": "running",
        "output_file": str(args.output_file),
        "requested_tasks": str(len(tasks)),
        "completed_tasks": str(len(completed)),
        "valid_judgments": str(sum(row.get("judgment") in {0, 1, 2, 3} for row in completed.values())),
        "invalid_tasks": str(sum(row.get("status") == "invalid" for row in completed.values())),
        "failed_tasks": str(sum(row.get("status") == "failed" for row in completed.values())),
        "started_at": started_at,
        "updated_at": timestamp(),
    }
    update_manifest(manifest, manifest_row)
    print(f"Output directory: {run_dir}", flush=True)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Jev UMBRELA evaluation: {len(tasks)} tasks", flush=True)
    print(f"Already completed: {len(completed)}; pending: {len(pending)}", flush=True)
    if args.metadata_text:
        print(f"Metadata prefix: {args.metadata_text}", flush=True)
    if args.metadata_map:
        print(f"Metadata map: {args.metadata_map}", flush=True)
    if args.query_title:
        print("Content injection: query as <h>title</h>", flush=True)
    if args.gov_injection:
        print("Content injection: synthetic Australian government-style provenance", flush=True)

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
            manifest_row.update(
                {
                    "completed_tasks": str(len(completed)),
                    "valid_judgments": str(sum(row.get("judgment") in {0, 1, 2, 3} for row in completed.values())),
                    "invalid_tasks": str(sum(row.get("status") == "invalid" for row in completed.values())),
                    "failed_tasks": str(sum(row.get("status") == "failed" for row in completed.values())),
                    "updated_at": timestamp(),
                }
            )
            update_manifest(manifest, manifest_row)

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
    manifest_row.update({
        "status": "finished",
        "completed_tasks": str(len(completed)),
        "valid_judgments": str(len(valid)),
        "invalid_tasks": str(sum(row.get("status") == "invalid" for row in completed.values())),
        "failed_tasks": str(sum(row.get("status") == "failed" for row in completed.values())),
        "updated_at": timestamp(),
    })
    update_manifest(manifest, manifest_row)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
