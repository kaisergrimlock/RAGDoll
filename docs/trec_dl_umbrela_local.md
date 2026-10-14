# Running UMBRELA on the local TREC-DL files

## Server environment

Run these evaluations in the existing `pyserini-v2` Conda environment, not
the `base` environment. It contains the compatible NumPy/SciPy versions
needed by RAGDoll and the TREC-DL tooling. The server does not currently have
`uv` installed, so invoke RAGDoll directly with Python and `PYTHONPATH`.

```bash
conda activate /research/remote/petabyte/users/s3891987/environments/pyserini-v2
cd /research/remote/petabyte/users/s3891987/LLM-Ranker-Attack/RAGDoll
export PYTHONPATH="$PWD/src"
```

Check the environment before a run:

```bash
python -c "import numpy, scipy, yaml; print(numpy.__version__, scipy.__version__)"
python -c "import boto3, tqdm; print('Bedrock and progress-bar dependencies ready')"
```

If the second check fails, install only the lightweight missing packages:

```bash
python -m pip install boto3
python -m pip install --no-deps tqdm
```

Avoid `pip install -e .` on this server: it may try to compile a newer
NumPy/SciPy release with the system's old GCC compiler.

## Organized Bedrock results

Use `tools/run_umbrela_bedrock.py` for Bedrock runs. It stores results by
model, attack condition, and year:

```text
results/umbrela/gpt-oss-20b/default/2021/
├── judgments.jsonl
└── raw-events/
```

Each model directory also contains `manifest.csv`, which records the status
of every condition/year run and the number of completed tasks. The launcher
uses resume mode by default; pass `--overwrite` only for a new run.

Example for the current default condition:

```bash
python tools/run_umbrela_bedrock.py \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --year 2021 \
  --condition default \
  --model gpt-oss-20b \
  --max-tokens 2048 \
  --max-concurrency 2 \
  --overwrite
```

If interrupted, rerun the same command without `--overwrite`:

```bash
python tools/run_umbrela_bedrock.py \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --year 2021 \
  --condition default \
  --model gpt-oss-20b \
  --max-tokens 2048 \
  --max-concurrency 2
```

The copied files in `unjudged/` are one row per query/passage pair. Convert one
or more years into RAGDoll's UMBRELA request format and matching gold files
with:

```bash
cd /research/remote/petabyte/users/s3891987/LLM-Ranker-Attack/RAGDoll
python tools/prepare_trec_dl_umbrela.py \
  --years 2019 2022 2023
```

The converter supports both the older 2019 CSV schema and the later `qid`/`pid`
schema. It writes `results/trec_dl_<year>.umbrela.jsonl` and
`results/gold/trec_dl/<year>/gold.csv`. The request JSONL groups all passages
for each query. The gold CSV retains the original relevance label and can be
joined to RAGDoll's output using `qid` and `docid`. Existing outputs are
preserved unless `--overwrite` is supplied.

Run a small materialization smoke test first:

```bash
PYTHONPATH="$PWD/src" python -m ragdoll.cli materialize umbrela \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.tasks.jsonl \
  --prompt-type bing
```

Then judge with UMBRELA:

```bash
PYTHONPATH="$PWD/src" python -m ragdoll.cli umbrela judge \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.judgments.jsonl \
  --raw-events-dir results/trec_dl_2021.raw-events \
  --include-trace \
  --overwrite
```

For long runs, use `--overwrite` only on the first invocation. Every completed
task is appended to the output JSONL immediately. If the process stops, rerun
the same command without `--overwrite` and add `--resume`; completed task IDs
are skipped and failed or unfinished tasks are retried:

```bash
PYTHONPATH="$PWD/src" python -m ragdoll.cli umbrela judge \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.gpt-oss-20b.judgments.jsonl \
  --raw-events-dir results/trec_dl_2021.gpt-oss-20b.raw-events \
  --provider bedrock \
  --model openai.gpt-oss-20b-1:0 \
  --max-tokens 2048 \
  --max-concurrency 2 \
  --resume
```

Do not combine `--resume` with `--overwrite`: overwrite intentionally deletes
the existing checkpoint/output first.

The current copied directory contains 2019, 2021, 2022, and 2023 files. A
2020 CSV still needs to be copied before running the 2020 evaluation. The
2019 file has no explicit `qid` column, so the converter creates a stable
query ID from the query text; files with `qid` retain their original IDs.

## Run UMBRELA through Jev

RAGDoll's built-in `ragdoll umbrela judge` uses its Pi agent runner. For Jev,
use `tools/run_jev_umbrela.py`. It uses a four-way TypeSafe `Choice` question
with the original UMBRELA grades 0--3, batches requests, shows a progress bar,
and resumes from an existing JSONL output. Jev results use the same organized
layout, with `jev` as the model/evaluator directory:

```text
results/umbrela/jev/
├── manifest.csv
└── default/2021/
    └── judgments.jsonl
```

The corresponding gold labels are stored separately for easy reuse:

```text
results/gold/trec_dl/2021/gold.csv
```

Install the SDK in the environment used to run the command:

```bash
pip install typesafe-sdk tqdm
```

Run a 100-task smoke test:

```bash
python tools/run_jev_umbrela.py \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --year 2021 \
  --condition default \
  --model jev \
  --limit 100 \
  --batch-size 16
```

Run all converted tasks:

```bash
python tools/run_jev_umbrela.py \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --year 2021 \
  --condition default \
  --model jev \
  --batch-size 16
```

The output retains RAGDoll-compatible `qid`, `docid`, `passage`, and
`judgment` fields, with Jev choice and confidence fields added. Re-running the
same command resumes completed tasks unless `--overwrite` is supplied.

Score Jev against the original relevance labels with:

```bash
python scoring/score_jev_umbrela.py \
  --gold-file results/gold/trec_dl/2021/gold.csv \
  --judgments-file results/umbrela/jev/default/2021/judgments.jsonl
```

This reports overall MAE, query-macro-averaged MAE, exact accuracy, MAE by
gold grade, and a gold-versus-Jev confusion table.

### Test author-gender metadata sensitivity

To reproduce the paper-style metadata perturbation, prefix every passage with
an author-gender statement while keeping the query, passage text, and Jev
relevance criteria unchanged:

```bash
python tools/run_jev_gender_metadata.py \
  --years 2019 2022 2023 \
  --gender male \
  --batch-size 16
```

The default sentence is:

```text
The author of this passage is male.
```

The results are stored separately from clean runs:

```text
results/umbrela/jev/metadata/gender/male/<year>/judgments.jsonl
```

Use `--gender female`, `--gender non-binary`, or
`--gender prefer-not-to-say` for additional conditions. To score one
condition against the original relevance judgments:

```bash
python scoring/score_jev_umbrela.py \
  --gold-file results/gold/trec_dl/2021/gold.csv \
  --judgments-file results/umbrela/jev/metadata/gender/male/2021/judgments.jsonl
```

This measures whether metadata changes relevance judgments; it is not a
prompt-injection or passage-ranking attack.

### Run the remaining years and update MAE

To run the years other than the existing 2021 run, use the multi-year wrapper.
It runs each year sequentially, preserves the normal resumable Jev outputs,
and upserts the resulting MAE into `results/eval/mae.csv`:

```bash
python tools/run_jev_all_years.py \
  --years 2019 2020 2022 2023 \
  --model jev \
  --condition default \
  --batch-size 16
```

The expected files are:

```text
results/trec_dl_<year>.umbrela.jsonl
results/gold/trec_dl/<year>/gold.csv
```

Missing years are reported and skipped by default. Add `--require-all` to stop
instead when any requested year has not yet been converted or copied. Use
`--overwrite` only when intentionally restarting completed years; otherwise
the wrapper resumes from the existing year-specific judgment files.

## Run UMBRELA through Bedrock GPT-OSS-20B

The built-in RAGDoll UMBRELA judge also supports Amazon Bedrock Converse.
Use the server's AWS CLI profile or its IAM/STS environment; do not put keys
on the command line. Bedrock's GPT-OSS-20B model ID is
`openai.gpt-oss-20b-1:0`.

Verify credentials and model access in the selected region:

```bash
aws sts get-caller-identity --region us-east-1
aws bedrock list-foundation-models --region us-east-1 \
  --by-provider openai --output json
```

For temporary credentials, export them in the same shell before running the
evaluation. Keep the values private and do not commit them:

```bash
export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'
export AWS_SESSION_TOKEN='...'
export AWS_REGION='us-east-1'
export AWS_DEFAULT_REGION="$AWS_REGION"
```

The project includes `boto3` as a runtime dependency. If this checkout has
not been synchronized since the Bedrock support was added, install `boto3`
in the active environment before the smoke test.

Run a 10-task smoke test:

```bash
PYTHONPATH="$PWD/src" python -m ragdoll.cli umbrela judge \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.gpt-oss-20b.smoke.judgments.jsonl \
  --raw-events-dir results/trec_dl_2021.gpt-oss-20b.smoke.raw-events \
  --provider bedrock \
  --model openai.gpt-oss-20b-1:0 \
  --max-tokens 2048 --max-concurrency 2 --limit 10 --overwrite
```

For the full run, remove `--limit 10`; use `--resume` if a run is
interrupted:

```bash
PYTHONPATH="$PWD/src" python -m ragdoll.cli umbrela judge \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --output-file results/trec_dl_2021.gpt-oss-20b.judgments.jsonl \
  --raw-events-dir results/trec_dl_2021.gpt-oss-20b.raw-events \
  --provider bedrock \
  --model openai.gpt-oss-20b-1:0 \
  --max-tokens 2048 --max-concurrency 4 --resume
```

### Qwen3-32B

Qwen3-32B uses the Bedrock model ID `qwen.qwen3-32b-v1:0`. The organized
launcher stores this run under `results/umbrela/qwen3-32b/default/2021/` and
updates its manifest:

```bash
python tools/run_umbrela_bedrock.py \
  --input-file results/trec_dl_2021.umbrela.jsonl \
  --year 2021 \
  --condition default \
  --model qwen3-32b \
  --bedrock-model qwen.qwen3-32b-v1:0 \
  --max-tokens 2048 \
  --max-concurrency 2
```
