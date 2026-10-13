# Running UMBRELA on the local TREC-DL files

The copied files in `unjudged/` are one row per query/passage pair. Convert a
year into RAGDoll's UMBRELA request format with:

```bash
cd /research/remote/petabyte/users/s3891987/LLM-Ranker-Attack/RAGDoll
python tools/trec_dl_csv_to_umbrela.py \
  unjudged/trec_dl_2021.csv \
  --output-file results/trec_dl_2021.umbrela.jsonl \
  --gold-file results/trec_dl_2021.gold.csv
```

The request JSONL groups all passages for each query. The gold CSV retains the
original relevance label and can be joined to RAGDoll's output using `qid` and
`docid`.

Run a small materialization smoke test first:

```bash
uv run ragdoll materialize umbrela \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.tasks.jsonl \
  --prompt-type bing
```

Then judge with UMBRELA:

```bash
uv run ragdoll umbrela judge \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.judgments.jsonl \
  --raw-events-dir results/trec_dl_2021.raw-events \
  --include-trace \
  --overwrite
```

The current copied directory contains 2019, 2021, 2022, and 2023 files. A
2020 CSV still needs to be copied before running the 2020 evaluation. The
2019 file has no explicit `qid` column, so the converter creates a stable
query ID from the query text; files with `qid` retain their original IDs.

## Run UMBRELA through Jev

RAGDoll's built-in `ragdoll umbrela judge` uses its Pi agent runner. For Jev,
use `tools/run_jev_umbrela.py`. It uses a four-way TypeSafe `Choice` question
with the original UMBRELA grades 0--3, batches requests, shows a progress bar,
and resumes from an existing JSONL output.

Install the SDK in the environment used to run the command:

```bash
pip install typesafe-sdk tqdm
```

Run a 100-task smoke test:

```bash
python tools/run_jev_umbrela.py \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.jev.judgments.jsonl \
  --limit 100 \
  --batch-size 16
```

Run all converted tasks:

```bash
python tools/run_jev_umbrela.py \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.jev.judgments.jsonl \
  --batch-size 16
```

The output retains RAGDoll-compatible `qid`, `docid`, `passage`, and
`judgment` fields, with Jev choice and confidence fields added. Re-running the
same command resumes completed tasks unless `--overwrite` is supplied.
