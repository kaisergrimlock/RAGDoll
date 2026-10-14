#!/usr/bin/env python3
"""Run a Bedrock UMBRELA evaluation with organized outputs and a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "model",
    "bedrock_model_id",
    "condition",
    "year",
    "status",
    "output_file",
    "raw_events_dir",
    "requested_tasks",
    "completed_tasks",
    "started_at",
    "updated_at",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-") or "unnamed"


def task_count(path: Path) -> int:
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = record.get("candidates", [])
            total += len(candidates) if isinstance(candidates, list) else 0
    return total


def completed_count(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "completed":
                total += 1
    return total


def write_manifest(path: Path, row: dict[str, str]) -> None:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    key = (row["condition"], row["year"])
    replaced = False
    for index, existing in enumerate(rows):
        if (existing.get("condition"), existing.get("year")) == key:
            rows[index] = {field: row.get(field, existing.get(field, "")) for field in FIELDS}
            replaced = True
            break
    if not replaced:
        rows.append({field: row.get(field, "") for field in FIELDS})
    rows.sort(key=lambda item: (item.get("condition", ""), item.get("year", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--year", required=True, help="Dataset year, e.g. 2019 or 2021")
    parser.add_argument("--condition", default="default", help="Attack condition, e.g. default, doh, dch, qi, keyword")
    parser.add_argument("--model", default="gpt-oss-20b", help="Output model label")
    parser.add_argument("--bedrock-model", default="openai.gpt-oss-20b-1:0")
    parser.add_argument("--results-root", type=Path, default=Path("results/umbrela"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-type", choices=["bing", "basic"], default="bing")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true", help="Resume; this is the default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.results_root / safe_name(args.model)
    run_dir = model_dir / safe_name(args.condition) / safe_name(args.year)
    output_file = run_dir / "judgments.jsonl"
    raw_events_dir = run_dir / "raw-events"
    manifest = model_dir / "manifest.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "model": args.model,
        "bedrock_model_id": args.bedrock_model,
        "condition": args.condition,
        "year": args.year,
        "status": "running",
        "output_file": str(output_file),
        "raw_events_dir": str(raw_events_dir),
        "requested_tasks": str(task_count(args.input_file)),
        "completed_tasks": str(completed_count(output_file)),
        "started_at": now(),
        "updated_at": now(),
    }
    write_manifest(manifest, row)

    command = [
        sys.executable,
        "-m",
        "ragdoll.cli",
        "umbrela",
        "judge",
        "--input-file",
        str(args.input_file),
        "--output-file",
        str(output_file),
        "--raw-events-dir",
        str(raw_events_dir),
        "--provider",
        "bedrock",
        "--model",
        args.bedrock_model,
        "--prompt-type",
        args.prompt_type,
        "--max-tokens",
        str(args.max_tokens),
        "--max-concurrency",
        str(args.max_concurrency),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    command.append("--overwrite" if args.overwrite else "--resume")

    env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(source_root / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    print("Output directory:", run_dir, flush=True)
    print("Manifest:", manifest, flush=True)
    print("Running:", " ".join(command), flush=True)
    try:
        completed = subprocess.run(command, cwd=source_root, env=env, check=False)
        status = "finished" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
    except KeyboardInterrupt:
        status = "interrupted"
        returncode = 130

    row["status"] = status
    row["completed_tasks"] = str(completed_count(output_file))
    row["updated_at"] = now()
    write_manifest(manifest, row)
    print(f"Manifest updated: status={status}, completed_tasks={row['completed_tasks']}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
