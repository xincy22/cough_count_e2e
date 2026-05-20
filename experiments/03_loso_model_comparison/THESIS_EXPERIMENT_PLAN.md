# Thesis Counting Experiment Plan

## Goal

Build the minimum defensible evidence package for the thesis cough-counting chapter:

1. Show that the original WST sliding-window cough detector is sensitive to cough events but is not a reliable event counter.
2. Evaluate an end-to-end cough counting model under subject-independent LOSO validation.
3. Use ablation to justify the final `TCN+GRU` structure against `TCN` and `GRU` baselines.

## Experiments

| Block | Model / Method | Status | Thesis role |
|---|---|---|---|
| WST window detector | sliding window + positive-window merge | Completed separately | Motivates why event counting needs a dedicated model |
| Main counting model | `M2 = TCN+GRU` | Running on remote RTX 4090 | Main result |
| Ablation baseline | `M1 = TCN` | Scheduled after `M2` in current remote run | Remove GRU |
| Ablation baseline | `M0 = GRU` | Scheduled by remote watcher after current run exits | Remove TCN |

The current `experiment.yaml` intentionally keeps only `M0/M1/M2` to avoid
accidentally restarting the old 8-model comparison.

## Metrics

Primary table fields:

| Field | Meaning |
|---|---|
| `mean_count_mae` | Mean count MAE across 15 LOSO folds |
| `std_count_mae` | Fold-to-fold standard deviation of count MAE |
| `mean_count_mae_pos` | Count MAE on positive cough samples |
| `mean_count_mae_neg` | Count MAE on negative samples |

Fold-level evidence is preserved in every `fold_XX_test_<subject>/test_results.json`.

## Output Artifacts

After all three models finish, run:

```bash
cd experiments/03_loso_model_comparison
python scripts/05_thesis_ablation_report.py
```

Expected outputs:

| Artifact | Purpose |
|---|---|
| `result/thesis_ablation_<timestamp>/ablation_summary.csv` | Paper table source |
| `result/thesis_ablation_<timestamp>/ablation_fold_details.csv` | Fold-level appendix / audit source |
| `result/thesis_ablation_<timestamp>/thesis_ablation_report.md` | Thesis-ready draft text and Markdown tables |

## Thesis Interpretation Template

The ablation experiment compares a GRU-only temporal model, a TCN-only model,
and a combined TCN+GRU model under the same LOSO protocol. The TCN module is
used to extract local temporal acoustic patterns, while the GRU module models
longer-range temporal context. If `TCN+GRU` obtains lower Count MAE than both
single-module baselines, the result supports using the combined structure as
the final cough-counting model.
