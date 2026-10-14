# Scoring scripts

Place scripts that compare model outputs with gold labels or calculate
evaluation metrics in this directory.

Current scripts:

- `score_jev_umbrela.py`: computes MAE, exact accuracy, per-grade MAE, and a
  gold-versus-Jev confusion table for UMBRELA judgments.
- `score_jev_mean_signed_error.py`: computes mean signed error (`judgment -
  gold`) for every Jev condition and the paired change relative to the clean
  Jev default.
- `test_jev_metadata_wilcoxon.py`: applies paired Wilcoxon signed-rank tests
  to signed and absolute error changes for the gender metadata conditions.

Example:

```bash
python scoring/score_jev_umbrela.py \
  --gold-file results/gold/trec_dl/2021/gold.csv \
  --judgments-file results/umbrela/jev/default/2021/judgments.jsonl
```

Generate the Jev-only signed-error comparison table:

```bash
python scoring/score_jev_mean_signed_error.py
```

The output is written to `results/eval/jev_mean_signed_error.csv`.

Run the paired tests with the SciPy-enabled server environment:

```bash
/research/remote/petabyte/users/s3891987/environments/pyserini-v2/bin/python \
  scoring/test_jev_metadata_wilcoxon.py
```

The output is written to `results/eval/jev_metadata_wilcoxon.csv`.
