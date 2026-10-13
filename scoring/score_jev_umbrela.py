#!/usr/bin/env python3
"""Compute MAE between Jev UMBRELA judgments and original relevance labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--judgments-file", type=Path, required=True)
    args = parser.parse_args()

    gold: dict[tuple[str, str], int] = {}
    with args.gold_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gold[(str(row["qid"]), str(row["docid"]))] = int(row["original_relevance"])

    errors: list[float] = []
    by_qid: defaultdict[str, list[float]] = defaultdict(list)
    by_grade: defaultdict[int, list[float]] = defaultdict(list)
    confusion: Counter[tuple[int, int]] = Counter()
    missing_gold = 0
    invalid = 0
    with args.judgments_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = row.get("judgment")
            key = (str(row.get("qid")), str(row.get("docid")))
            if key not in gold:
                missing_gold += 1
                continue
            if not isinstance(prediction, int) or prediction not in range(4):
                invalid += 1
                continue
            target = gold[key]
            error = abs(prediction - target)
            errors.append(error)
            by_qid[key[0]].append(error)
            by_grade[target].append(error)
            confusion[(target, prediction)] += 1

    mae = sum(errors) / len(errors) if errors else None
    print(json.dumps({
        "gold_file": str(args.gold_file),
        "judgments_file": str(args.judgments_file),
        "matched": len(errors),
        "missing_gold": missing_gold,
        "invalid_judgments": invalid,
        "mae": round(mae, 6) if mae is not None else None,
        "exact_accuracy": round(sum(v for (g, p), v in confusion.items() if g == p) / len(errors), 6) if errors else None,
        "mae_by_gold_grade": {
            str(grade): round(sum(values) / len(values), 6)
            for grade, values in sorted(by_grade.items())
        },
        "mean_query_mae": round(sum(sum(values) / len(values) for values in by_qid.values()) / len(by_qid), 6) if by_qid else None,
        "confusion_gold_rows": {
            str(grade): {str(prediction): confusion[(grade, prediction)] for prediction in range(4)}
            for grade in range(4)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
