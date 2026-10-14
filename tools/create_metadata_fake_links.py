#!/usr/bin/env python3
"""Create deterministic fake metadata links for every unique dataset query.

The generated URLs are metadata stimuli for experiments, not real sources or
retrieval targets. They intentionally use realistic-looking domains and
query-derived fake paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path


def stable_qid(query: str) -> str:
    return f"q-{hashlib.sha1(query.encode('utf-8')).hexdigest()[:16]}"


def slugify(value: str, limit: int = 96) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return (slug or "query")[:limit].rstrip("-")


def read_queries(path: Path) -> list[dict[str, str]]:
    queries: OrderedDict[str, dict[str, str]] = OrderedDict()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "query" not in fields:
            raise ValueError(f"{path} does not contain a query column")
        for row in reader:
            query = (row.get("query") or "").strip()
            if not query:
                continue
            qid = (row.get("qid") or "").strip() or stable_qid(query)
            queries.setdefault(qid, {"qid": qid, "query": query})
    return list(queries.values())


def write_links(path: Path, rows: list[dict[str, str]], source_type: str, domain: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            slug = slugify(row["query"])
            suffix = row["qid"].removeprefix("q-")
            handle.write(
                json.dumps(
                    {
                        **row,
                        "source_type": source_type,
                        "fake_link": f"https://{domain}/{slug}-{suffix}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("unjudged"))
    parser.add_argument("--output-dir", type=Path, default=Path("attack/metadata/links"))
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    years = args.years or sorted(
        int(match.group(1))
        for path in input_dir.glob("trec_dl_*.csv")
        if (match := re.fullmatch(r"trec_dl_(\d{4})\.csv", path.name))
    )

    total = 0
    for year in years:
        input_path = input_dir / f"trec_dl_{year}.csv"
        if not input_path.exists():
            print(f"SKIP {year}: missing {input_path}", flush=True)
            continue
        year_dir = output_dir / str(year)
        wikipedia_path = year_dir / "wikipedia_links.jsonl"
        low_trust_path = year_dir / "low_trust_links.jsonl"
        if not args.overwrite and (wikipedia_path.exists() or low_trust_path.exists()):
            print(f"SKIP {year}: output exists; use --overwrite to regenerate", flush=True)
            continue

        queries = read_queries(input_path)
        write_links(wikipedia_path, queries, "wikipedia-style", "en.wikipedia.org/wiki")
        write_links(low_trust_path, queries, "low-trust", "daily-world-report.com")
        total += len(queries)
        print(json.dumps({"year": year, "queries": len(queries), "output": str(year_dir)}, indent=2), flush=True)

    print(f"Generated fake links for {total} unique queries", flush=True)


if __name__ == "__main__":
    main()
