#!/usr/bin/env python3
"""Test paired Jev metadata changes against clean default.

For each year and gender, this applies a two-sided paired Wilcoxon signed-rank
test to per-document changes in signed error and absolute error. Holm
correction is applied separately within each metric family.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import wilcoxon


FIELDS = [
    "metric",
    "year",
    "gender",
    "n_pairs",
    "clean_mean",
    "metadata_mean",
    "mean_change",
    "wilcoxon_statistic",
    "p_value",
    "holm_p_value",
    "significant_holm_0_05",
]


def load_gold(path: Path) -> dict[tuple[str, str], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (str(row["qid"]), str(row["docid"])): int(row["original_relevance"])
            for row in csv.DictReader(handle)
        }


def load_judgments(path: Path) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            judgment = row.get("judgment")
            if isinstance(judgment, int) and judgment in range(4):
                result[(str(row.get("qid")), str(row.get("docid")))] = judgment
    return result


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/umbrela/jev"))
    parser.add_argument("--gold-root", type=Path, default=Path("results/gold/trec_dl"))
    parser.add_argument("--output", type=Path, default=Path("results/eval/jev_metadata_wilcoxon.csv"))
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2021, 2022, 2023])
    parser.add_argument("--genders", nargs="+", default=["male", "female"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = args.results_root if args.results_root.is_absolute() else root / args.results_root
    gold_root = args.gold_root if args.gold_root.is_absolute() else root / args.gold_root
    output = args.output if args.output.is_absolute() else root / args.output

    pending: list[dict] = []
    for year in args.years:
        clean_path = results_root / "default" / str(year) / "judgments.jsonl"
        gold_path = gold_root / str(year) / "gold.csv"
        if not clean_path.exists() or not gold_path.exists():
            print(f"SKIP {year}: missing clean judgments or gold", flush=True)
            continue
        clean = load_judgments(clean_path)
        gold = load_gold(gold_path)
        for gender in args.genders:
            metadata_path = results_root / "metadata" / "gender" / gender / str(year) / "judgments.jsonl"
            if not metadata_path.exists():
                print(f"SKIP {year}/{gender}: metadata judgments missing", flush=True)
                continue
            metadata = load_judgments(metadata_path)
            keys = set(clean) & set(metadata) & set(gold)
            clean_signed = [clean[key] - gold[key] for key in keys]
            metadata_signed = [metadata[key] - gold[key] for key in keys]
            clean_absolute = [abs(value) for value in clean_signed]
            metadata_absolute = [abs(value) for value in metadata_signed]
            for metric, before, after in (
                ("signed_error", clean_signed, metadata_signed),
                ("absolute_error", clean_absolute, metadata_absolute),
            ):
                if len(keys) < 2:
                    continue
                result = wilcoxon(after, before, alternative="two-sided", zero_method="wilcox", method="auto")
                pending.append(
                    {
                        "metric": metric,
                        "year": str(year),
                        "gender": gender,
                        "n_pairs": len(keys),
                        "clean_mean": sum(before) / len(before),
                        "metadata_mean": sum(after) / len(after),
                        "mean_change": (sum(after) - sum(before)) / len(before),
                        "wilcoxon_statistic": float(result.statistic),
                        "p_value": float(result.pvalue),
                    }
                )

    for metric in {row["metric"] for row in pending}:
        group = [row for row in pending if row["metric"] == metric]
        adjusted = holm([row["p_value"] for row in group])
        for row, adjusted_p in zip(group, adjusted):
            row["holm_p_value"] = adjusted_p
            row["significant_holm_0_05"] = adjusted_p < 0.05

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in pending:
            writer.writerow(
                {
                    **row,
                    "clean_mean": fmt(row["clean_mean"]),
                    "metadata_mean": fmt(row["metadata_mean"]),
                    "mean_change": fmt(row["mean_change"]),
                    "wilcoxon_statistic": fmt(row["wilcoxon_statistic"]),
                    "p_value": fmt(row["p_value"]),
                    "holm_p_value": fmt(row["holm_p_value"]),
                }
            )
    print(f"Wrote {output} ({len(pending)} tests)", flush=True)


if __name__ == "__main__":
    main()
