# Local TREC-DL data for UMBRELA

These CSV files contain one query/passage pair per row:

```text
query or qid, passage identifier, passage text, original relevance label
```

The current folder contains:

```text
trec_dl_2019.csv  9,260 passages / 43 query groups
trec_dl_2021.csv  2,475 rows / 54 queries
trec_dl_2022.csv  2,862 rows / 76 queries
trec_dl_2023.csv  2,460 rows / 82 queries
```

The 2019 file has no explicit `qid`; the converter creates a stable ID from
the query text. The 2021–2023 files retain their original `qid` values. A
2020 file has not yet been copied into this directory.

## Convert a year for RAGDoll

From the RAGDoll directory:

```bash
python tools/trec_dl_csv_to_umbrela.py \
  unjudged/trec_dl_2021.csv \
  --output-file results/trec_dl_2021.umbrela.jsonl \
  --gold-file results/trec_dl_2021.gold.csv
```

This creates:

- `*.umbrela.jsonl`: one UMBRELA request per query, with all passages as candidates;
- `*.gold.csv`: the original relevance labels, for comparison with RAGDoll judgments.

Materialize prompts before spending model calls:

```bash
uv run ragdoll materialize umbrela \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.tasks.jsonl \
  --prompt-type bing
```

Run the evaluation with:

```bash
uv run ragdoll umbrela judge \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.judgments.jsonl \
  --raw-events-dir results/trec_dl_2021.raw-events \
  --include-trace \
  --overwrite
```

The judgment output can be joined to the gold CSV on `qid` and `docid`.
The converter and the longer explanation are in
`../tools/trec_dl_csv_to_umbrela.py` and `../docs/trec_dl_umbrela_local.md`.
