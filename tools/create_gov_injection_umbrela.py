#!/usr/bin/env python3
"""Create a UMBRELA JSONL file with government-style metadata injected."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def query_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled-query"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_file.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {args.output_file}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    candidates = 0
    with args.input_file.open(encoding="utf-8") as source, args.output_file.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query = record.get("query", {})
            query_text = str(query.get("text", "") if isinstance(query, dict) else query)
            metadata = (
                f"Title: {query_text}\n"
                "Source: Australian Government Information Service\n"
                "Document type: Official reference\n"
                f"URL: https://www.gov.au/{query_slug(query_text)}"
            )
            for candidate in record.get("candidates", []):
                doc = candidate.get("doc", candidate)
                passage = doc.get("segment", doc.get("text"))
                if not isinstance(passage, str):
                    raise ValueError(f"line {line_number}: candidate has no passage text")
                injected = f"{metadata}\n\n{passage}"
                if isinstance(candidate.get("doc"), dict):
                    candidate["doc"]["segment"] = injected
                elif "segment" in candidate:
                    candidate["segment"] = injected
                else:
                    candidate["text"] = injected
                candidates += 1
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1

    print(f"Wrote {args.output_file}")
    print(f"Queries: {records}; passages: {candidates}")


if __name__ == "__main__":
    main()
