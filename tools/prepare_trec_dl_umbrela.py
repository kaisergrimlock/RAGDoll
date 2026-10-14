#!/usr/bin/env python3
"""Convert TREC-DL passage CSV files into grouped UMBRELA requests and gold CSVs.

Supported input schemas are either ``qid,query,pid,passage,relevance`` or the
older ``query,docid,passage,relevance`` form. For files without qids, a stable
query hash is generated so the request and gold files use matching IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path


def stable_qid(query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    return f"q-{digest}"


def convert(input_path: Path, request_path: Path, gold_path: Path) -> dict[str, int | str]:
    grouped: OrderedDict[str, dict] = OrderedDict()
    gold_rows: list[dict[str, str]] = []
    rows_read = 0

    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query", "passage", "relevance"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_path} is missing columns: {', '.join(sorted(missing))}")
        docid_column = "docid" if "docid" in (reader.fieldnames or []) else "pid"
        if docid_column not in (reader.fieldnames or []):
            raise ValueError(f"{input_path} needs either a docid or pid column")

        for line_number, row in enumerate(reader, start=2):
            rows_read += 1
            query = (row.get("query") or "").strip()
            docid = (row.get(docid_column) or "").strip()
            passage = row.get("passage") or ""
            if not query or not docid:
                raise ValueError(f"{input_path}:{line_number} has an empty query or document ID")
            try:
                relevance = int(row["relevance"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{input_path}:{line_number} has an invalid relevance value") from exc
            if relevance not in range(4):
                raise ValueError(f"{input_path}:{line_number} relevance must be 0, 1, 2, or 3")

            qid = (row.get("qid") or "").strip() or stable_qid(query)
            if qid not in grouped:
                grouped[qid] = {
                    "query": {"qid": qid, "text": query},
                    "candidates": [],
                }
            grouped[qid]["candidates"].append({"doc": {"docid": docid, "segment": passage}})
            gold_rows.append(
                {
                    "qid": qid,
                    "query": query,
                    "docid": docid,
                    "original_relevance": str(relevance),
                }
            )

    request_path.parent.mkdir(parents=True, exist_ok=True)
    with request_path.open("w", encoding="utf-8") as handle:
        for record in grouped.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "query", "docid", "original_relevance"])
        writer.writeheader()
        writer.writerows(gold_rows)

    return {
        "input": str(input_path),
        "requests": str(request_path),
        "gold": str(gold_path),
        "rows": rows_read,
        "queries": len(grouped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--input-template", default="unjudged/trec_dl_{year}.csv")
    parser.add_argument("--request-template", default="results/trec_dl_{year}.umbrela.jsonl")
    parser.add_argument("--gold-template", default="results/gold/trec_dl/{year}/gold.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    converted: list[dict[str, int | str]] = []
    for year in args.years:
        paths = {
            "input": Path(args.input_template.format(year=year)),
            "request": Path(args.request_template.format(year=year)),
            "gold": Path(args.gold_template.format(year=year)),
        }
        for key, path in paths.items():
            if not path.is_absolute():
                paths[key] = repo_root / path
        if not paths["input"].exists():
            print(f"SKIP {year}: input missing: {paths['input']}", flush=True)
            continue
        if not args.overwrite and (paths["request"].exists() or paths["gold"].exists()):
            print(f"SKIP {year}: output already exists; use --overwrite to replace it", flush=True)
            continue
        result = convert(paths["input"], paths["request"], paths["gold"])
        converted.append(result)
        print(json.dumps(result, indent=2), flush=True)

    print("Converted years:", json.dumps(converted, indent=2), flush=True)


if __name__ == "__main__":
    main()
