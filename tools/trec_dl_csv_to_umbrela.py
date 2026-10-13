#!/usr/bin/env python3
"""Convert TREC-DL query/passage CSV files to RAGDoll UMBRELA requests.

The input CSV is one row per query/passage pair and must contain query,
passage, and relevance columns. It may use qid/docid or pid for identifiers.
The generated gold CSV preserves the original relevance labels for joining
against RAGDoll's judgment output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path


def first_value(row: dict[str, str], names: tuple[str, ...], *, required: bool = True) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    if required:
        raise ValueError(f"missing one of columns: {', '.join(names)}")
    return None


def stable_query_id(query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    return f"query-{digest}"


def convert(input_file: Path, output_file: Path, gold_file: Path) -> tuple[int, int]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, object]]] = OrderedDict()
    gold_rows: list[dict[str, str]] = []

    with input_file.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{input_file} has no CSV header")
        for row_number, row in enumerate(reader, start=2):
            query = first_value(row, ("query", "text"))
            passage = first_value(row, ("passage", "segment", "doc"))
            qid = first_value(row, ("qid", "query_id"), required=False) or stable_query_id(query)
            docid = first_value(row, ("docid", "pid", "doc_id"), required=False)
            if docid is None:
                docid = f"row-{row_number}"
            relevance = first_value(row, ("relevance", "label", "grade"))
            key = (qid, query)
            grouped.setdefault(key, []).append(
                {"doc": {"docid": docid, "segment": passage}}
            )
            gold_rows.append(
                {"qid": qid, "query": query, "docid": docid, "original_relevance": relevance}
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    gold_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for (qid, query), candidates in grouped.items():
            handle.write(json.dumps({"query": {"qid": qid, "text": query}, "candidates": candidates}, ensure_ascii=False) + "\n")

    with gold_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "query", "docid", "original_relevance"])
        writer.writeheader()
        writer.writerows(gold_rows)

    return len(grouped), len(gold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--gold-file", type=Path)
    args = parser.parse_args()
    output = args.output_file or args.input_csv.with_suffix(".umbrela.jsonl")
    gold = args.gold_file or args.input_csv.with_suffix(".gold.csv")
    queries, passages = convert(args.input_csv, output, gold)
    print(f"queries={queries} passages={passages}")
    print(f"requests={output}")
    print(f"gold={gold}")


if __name__ == "__main__":
    main()
