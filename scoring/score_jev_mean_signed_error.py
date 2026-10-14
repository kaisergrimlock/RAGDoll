#!/usr/bin/env python3
"""Compare Jev mean signed error against gold and clean default.

Mean signed error is defined here as ``judgment - gold``. Positive values mean
that Jev over-judged relevance; negative values mean that it under-judged it.
For metadata conditions, the comparison with clean default is paired on the
same qid/document judgments whenever both runs are valid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "runid",
    "model",
    "condition",
    "year",
    "matched",
    "mean_signed_error_vs_gold",
    "clean_default_mean_signed_error",
    "delta_mean_signed_error_vs_clean_default",
    "mean_judgment_change_vs_clean_default",
]


def load_gold(path: Path) -> dict[tuple[str, str], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (str(row["qid"]), str(row["docid"])): int(row["original_relevance"])
            for row in csv.DictReader(handle)
        }


def load_valid(path: Path) -> dict[tuple[str, str], int]:
    values: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            judgment = row.get("judgment")
            if isinstance(judgment, int) and judgment in range(4):
                values[(str(row.get("qid")), str(row.get("docid")))] = judgment
    return values


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/umbrela/jev"))
    parser.add_argument("--gold-root", type=Path, default=Path("results/gold/trec_dl"))
    parser.add_argument("--output", type=Path, default=Path("results/eval/jev_mean_signed_error.csv"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = args.results_root if args.results_root.is_absolute() else root / args.results_root
    gold_root = args.gold_root if args.gold_root.is_absolute() else root / args.gold_root
    output = args.output if args.output.is_absolute() else root / args.output

    default_by_year: dict[str, dict[tuple[str, str], int]] = {}
    default_dir = results_root / "default"
    for path in sorted(default_dir.glob("*/judgments.jsonl")):
        default_by_year[path.parent.name] = load_valid(path)

    rows: list[dict[str, str]] = []
    for judgments_path in sorted(results_root.glob("**/judgments.jsonl")):
        relative = judgments_path.relative_to(results_root)
        if len(relative.parts) < 2:
            continue
        year = relative.parts[-2]
        condition_parts = relative.parts[:-2]
        condition = "/".join(condition_parts)
        if condition == "default":
            condition_label = "default"
        else:
            condition_label = condition

        gold_path = gold_root / year / "gold.csv"
        clean = default_by_year.get(year)
        if not gold_path.exists() or clean is None:
            print(f"SKIP {condition_label}/{year}: missing gold or clean default", flush=True)
            continue

        gold = load_gold(gold_path)
        current = load_valid(judgments_path)
        keys = set(current) & set(gold)
        current_errors = [current[key] - gold[key] for key in keys]
        clean_keys = keys & set(clean)
        clean_errors = [clean[key] - gold[key] for key in clean_keys]
        paired_current = [current[key] for key in clean_keys]
        paired_clean = [clean[key] for key in clean_keys]

        current_mse = mean(current_errors)
        clean_mse = mean(clean_errors)
        delta = current_mse - clean_mse if current_mse is not None and clean_mse is not None else None
        prediction_change = mean([current_value - clean_value for current_value, clean_value in zip(paired_current, paired_clean)])

        rows.append(
            {
                "runid": f"jev-{year}-{condition_label.replace('/', '-')}",
                "model": "jev",
                "condition": condition_label,
                "year": year,
                "matched": str(len(keys)),
                "mean_signed_error_vs_gold": fmt(current_mse),
                "clean_default_mean_signed_error": fmt(clean_mse),
                "delta_mean_signed_error_vs_clean_default": fmt(delta),
                "mean_judgment_change_vs_clean_default": fmt(prediction_change),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} ({len(rows)} Jev runs)", flush=True)


if __name__ == "__main__":
    main()
