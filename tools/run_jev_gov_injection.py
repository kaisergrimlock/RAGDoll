#!/usr/bin/env python3
"""Run Jev with synthetic government-style provenance injected into passages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--input-template", default="results/trec_dl_{year}.umbrela.jsonl")
    parser.add_argument("--results-root", type=Path, default=Path("results/umbrela"))
    parser.add_argument("--model", default="jev")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-probabilities", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runner = Path(__file__).with_name("run_jev_umbrela.py")
    results_root = args.results_root if args.results_root.is_absolute() else repo_root / args.results_root
    condition = "content/gov-injection"

    for year in args.years:
        input_path = Path(args.input_template.format(year=year))
        if not input_path.is_absolute():
            input_path = repo_root / input_path
        if not input_path.exists():
            print(f"SKIP {year}: input missing: {input_path}", flush=True)
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
            condition,
            "--year",
            str(year),
            "--batch-size",
            str(args.batch_size),
            "--gov-injection",
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.overwrite:
            command.append("--overwrite")
        if args.include_probabilities:
            command.append("--include-probabilities")

        print(f"\n=== Starting gov-injection Jev evaluation for {year} ===", flush=True)
        print(f"Output condition: {condition}", flush=True)
        print("Injection: government-style title, source, document type, and gov.au URL", flush=True)
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"Jev gov-injection evaluation failed for {year} "
                f"with exit code {result.returncode}"
            )


if __name__ == "__main__":
    main()
