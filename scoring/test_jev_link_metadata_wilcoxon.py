#!/usr/bin/env python3
"""Compare Jev judgments produced with Wikipedia-style and low-trust links.

The comparison is paired by qid/document, so each test uses the same judged
items under both link conditions.  Tests are run on signed error and absolute
error relative to the TREC gold labels.  Holm correction is applied across all
tests in the output.
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
    "wikipedia_condition",
    "low_trust_condition",
    "n_pairs",
    "wikipedia_mean",
    "low_trust_mean",
    "mean_change_low_trust_minus_wikipedia",
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
    parser.add_argument("--output", type=Path, default=Path("results/eval/jev_link_metadata_wilcoxon.csv"))
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2021, 2022, 2023])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = args.results_root if args.results_root.is_absolute() else root / args.results_root
    gold_root = args.gold_root if args.gold_root.is_absolute() else root / args.gold_root
    output = args.output if args.output.is_absolute() else root / args.output

    rows: list[dict] = []
    for year in args.years:
        wikipedia_path = results_root / "metadata" / "links" / "wikipedia" / str(year) / "judgments.jsonl"
        low_trust_path = results_root / "metadata" / "links" / "low_trust" / str(year) / "judgments.jsonl"
        gold_path = gold_root / str(year) / "gold.csv"
        if not wikipedia_path.exists() or not low_trust_path.exists() or not gold_path.exists():
            print(f"SKIP {year}: missing Wikipedia, low-trust, or gold file", flush=True)
            continue

        wikipedia = load_judgments(wikipedia_path)
        low_trust = load_judgments(low_trust_path)
        gold = load_gold(gold_path)
        keys = set(wikipedia) & set(low_trust) & set(gold)
        if len(keys) < 2:
            print(f"SKIP {year}: only {len(keys)} matched pairs", flush=True)
            continue

        wikipedia_signed = [wikipedia[key] - gold[key] for key in keys]
        low_trust_signed = [low_trust[key] - gold[key] for key in keys]
        wikipedia_absolute = [abs(value) for value in wikipedia_signed]
        low_trust_absolute = [abs(value) for value in low_trust_signed]

        for metric, before, after in (
            ("signed_error", wikipedia_signed, low_trust_signed),
            ("absolute_error", wikipedia_absolute, low_trust_absolute),
        ):
            result = wilcoxon(
                after,
                before,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            rows.append(
                {
                    "metric": metric,
                    "year": str(year),
                    "wikipedia_condition": "metadata/links/wikipedia",
                    "low_trust_condition": "metadata/links/low_trust",
                    "n_pairs": len(keys),
                    "wikipedia_mean": sum(before) / len(before),
                    "low_trust_mean": sum(after) / len(after),
                    "mean_change_low_trust_minus_wikipedia": (sum(after) - sum(before)) / len(before),
                    "wilcoxon_statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )

    adjusted = holm([row["p_value"] for row in rows]) if rows else []
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_p_value"] = adjusted_p
        row["significant_holm_0_05"] = adjusted_p < 0.05

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "wikipedia_mean": fmt(row["wikipedia_mean"]),
                    "low_trust_mean": fmt(row["low_trust_mean"]),
                    "mean_change_low_trust_minus_wikipedia": fmt(row["mean_change_low_trust_minus_wikipedia"]),
                    "wilcoxon_statistic": fmt(row["wilcoxon_statistic"]),
                    "p_value": fmt(row["p_value"]),
                    "holm_p_value": fmt(row["holm_p_value"]),
                }
            )
    print(f"Wrote {output} ({len(rows)} tests)", flush=True)


if __name__ == "__main__":
    main()
