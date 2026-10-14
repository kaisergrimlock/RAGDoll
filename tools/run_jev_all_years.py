#!/usr/bin/env python3
"""Run Jev UMBRELA evaluations for multiple TREC-DL years and record MAE.

The input JSONL files must already be in RAGDoll's grouped UMBRELA format and
the gold files must contain ``qid``, ``docid`` and ``original_relevance``.
Each year is delegated to ``run_jev_umbrela.py``, so runs remain resumable and
retain the normal progress bar and Jev manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MAE_FIELDS = ["runid", "model", "mae", "condition", "year"]


def calculate_mae(gold_path: Path, judgments_path: Path) -> tuple[float | None, int, int, int]:
    """Return MAE and matching/invalid/missing counts for one run."""
    gold: dict[tuple[str, str], int] = {}
    with gold_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gold[(str(row["qid"]), str(row["docid"]))] = int(row["original_relevance"])

    errors: list[int] = []
    invalid = 0
    missing_gold = 0
    with judgments_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("qid")), str(row.get("docid")))
            if key not in gold:
                missing_gold += 1
                continue
            prediction = row.get("judgment")
            if not isinstance(prediction, int) or prediction not in range(4):
                invalid += 1
                continue
            errors.append(abs(prediction - gold[key]))

    mae = sum(errors) / len(errors) if errors else None
    return mae, len(errors), invalid, missing_gold


def upsert_mae(path: Path, row: dict[str, str]) -> None:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    key = (row["runid"], row["model"], row["condition"], row["year"])
    rows = [
        existing
        for existing in rows
        if (existing.get("runid"), existing.get("model"), existing.get("condition"), existing.get("year")) != key
    ]
    rows.append({field: row.get(field, "") for field in MAE_FIELDS})
    rows.sort(key=lambda item: (item.get("model", ""), item.get("condition", ""), item.get("year", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2022, 2023])
    parser.add_argument("--input-template", default="results/trec_dl_{year}.umbrela.jsonl")
    parser.add_argument("--gold-template", default="results/gold/trec_dl/{year}/gold.csv")
    parser.add_argument("--results-root", type=Path, default=Path("results/umbrela"))
    parser.add_argument("--eval-file", type=Path, default=Path("results/eval/mae.csv"))
    parser.add_argument("--model", default="jev")
    parser.add_argument("--condition", default="default")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument("--require-all", action="store_true", help="Fail if any requested year's inputs are missing")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runner = Path(__file__).with_name("run_jev_umbrela.py")
    results_root = args.results_root if args.results_root.is_absolute() else repo_root / args.results_root
    eval_file = args.eval_file if args.eval_file.is_absolute() else repo_root / args.eval_file
    missing: list[str] = []
    completed: list[dict[str, object]] = []

    for year in args.years:
        input_path = Path(args.input_template.format(year=year))
        gold_path = Path(args.gold_template.format(year=year))
        if not input_path.is_absolute():
            input_path = repo_root / input_path
        if not gold_path.is_absolute():
            gold_path = repo_root / gold_path
        if not input_path.exists() or not gold_path.exists():
            missing.append(
                f"{year}: input={'present' if input_path.exists() else 'missing'}, "
                f"gold={'present' if gold_path.exists() else 'missing'}"
            )
            print(f"SKIP {year}: required input or gold file is missing", flush=True)
            continue

        command = [
            sys.executable,
            str(runner),
            "--input-file",
            str(input_path),
            "--results-root",
            str(results_root),
            "--model",
            args.model,
            "--condition",
            args.condition,
            "--year",
            str(year),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.overwrite:
            command.append("--overwrite")
        if args.include_probabilities:
            command.append("--include-probabilities")

        print(f"\n=== Starting Jev UMBRELA for {year} ===", flush=True)
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Jev evaluation failed for {year} with exit code {result.returncode}")

        judgment_path = results_root / args.model / args.condition / str(year) / "judgments.jsonl"
        mae, matched, invalid, missing_gold = calculate_mae(gold_path, judgment_path)
        if mae is None:
            print(f"WARNING {year}: no valid matched judgments; MAE not recorded", flush=True)
            continue
        runid = f"{args.model}-{year}-{args.condition}"
        upsert_mae(
            eval_file,
            {
                "runid": runid,
                "model": args.model,
                "mae": f"{mae:.6f}",
                "condition": args.condition,
                "year": str(year),
            },
        )
        result_row = {
            "year": year,
            "mae": round(mae, 6),
            "matched": matched,
            "invalid": invalid,
            "missing_gold": missing_gold,
            "judgments": str(judgment_path),
        }
        completed.append(result_row)
        print(json.dumps(result_row, indent=2), flush=True)
        print(f"Updated MAE CSV: {eval_file}", flush=True)

    if missing:
        print("\nMissing years:", flush=True)
        for item in missing:
            print(f"  {item}", flush=True)
        if args.require_all:
            raise SystemExit("One or more requested years are missing required files")

    print("\nFinished requested Jev years:", flush=True)
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == "__main__":
    main()
