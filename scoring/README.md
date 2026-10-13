# Scoring scripts

Place scripts that compare model outputs with gold labels or calculate
evaluation metrics in this directory.

Current scripts:

- `score_jev_umbrela.py`: computes MAE, exact accuracy, per-grade MAE, and a
  gold-versus-Jev confusion table for UMBRELA judgments.

Example:

```bash
python scoring/score_jev_umbrela.py \
  --gold-file results/trec_dl_2021.gold.csv \
  --judgments-file results/trec_dl_2021.jev.judgments.jsonl
```
