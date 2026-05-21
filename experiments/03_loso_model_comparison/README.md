# Experiment 03: LOSO 10-Model Comparison

This experiment compares cough-counting model families under one subject-independent protocol. It is a standalone capacity-controlled model comparison and not a strict component ablation.

## Goal

Compare 10 cough-counting models with the same data, LOSO split logic, training protocol, metric definitions, and an approximately 0.7M trainable-parameter budget:

```text
CNN1D / DSCNN / ResCNN / CRNN / BiCRNN / BiGRU / TCN / TCN-Attn / TCN+UniGRU / TCN+BiGRU
```

The experiment answers:

> Which model family is more suitable for end-to-end cough-event counting under leave-one-subject-out validation?

## Key Files

| File | Purpose |
|---|---|
| `experiment.yaml` | Default 10-model LOSO config for `scripts/03_loso.py` |
| `configs/structure_compare_v2_0p7m.yaml` | Formal 10-model config used by the queue runner |
| `configs/loso_10model_jobs_0p7m.yaml` | 4-GPU queue plan for the 10 jobs |
| `scripts/03_loso.py` | Run LOSO training/testing for one or more models |
| `scripts/06_loso_10model_queue.py` | Main audit/run/report entrypoint |
| `scripts/07_start_loso_10model_4gpu.sh` | Start four shard workers on a 4-GPU machine |
| `scripts/08_report_loso_10model.sh` | Generate the report after training finishes |

## Shared Protocol

| Item | Setting |
|---|---|
| Dataset | EdgeAI cough-counting dataset |
| Validation | 15-fold LOSO; one subject is held out for each test fold |
| Input feature | STFT log-magnitude, shape `[B, F, T]` |
| STFT | `win=1024`, `hop=256`, single-microphone frequency bins `F=513` |
| Microphones | `mic=both`; out/body microphone samples are both used, not channel-concatenated |
| Windowing | `window_sec=8.0`, `hop_sec=4.0` |
| Density target | `skewed_gaussian`, `sigma_left_sec=0.03`, `sigma_right_sec=0.10` |
| Epochs | `500` |
| Batch size | `24` |
| Dataloader workers | `4` |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=0` |
| LR schedule | cosine cycle, `lr_cycle_epochs=100`, `lr_eta_min=1e-8` |
| Checkpoint selection | best checkpoint per fold by `val_count_mae` |
| Primary metric | `test_count_mae` |
| Secondary metrics | `test_count_mae_pos`, `test_count_mae_neg` |

## Model Configurations

| ID | Model | Params | Summary |
|---|---|---:|---|
| S0 | CNN1D | 726,641 | 5 Conv1d layers, channels `[48,96,160,224,288]` |
| S1 | DSCNN | 699,974 | 5 depthwise-separable residual CNN blocks |
| S2 | ResCNN | 699,905 | 3 residual CNN blocks, channels `[64,192,256]` |
| S3 | CRNN | 700,001 | CNN `[96,128,128,144,176]` + unidirectional GRU hidden 256 |
| S4 | BiCRNN | 700,001 | CNN `[64,144,176,224,224]` + bidirectional GRU hidden 112 per direction |
| S5 | BiGRU | 703,969 | `1x1 Conv` projection 288 + bidirectional GRU hidden 192 per direction |
| S6 | TCN | 699,521 | 4 residual dilated TCN blocks, channels 160, dilation `1,2,4,8` |
| S7 | TCN-Attn | 692,609 | 5 TCN blocks + 1 single-head self-attention block |
| S8 | TCN+UniGRU | 699,297 | 4 TCN blocks, channels 128 + unidirectional GRU hidden 224 |
| S9 | TCN+BiGRU | 698,209 | 4 TCN blocks, channels 128 + bidirectional GRU hidden 144 per direction |

## Reproduce

From the repository root:

```bash
uv sync --locked
```

Prepare data:

```bash
python experiments/00_data_prep/scripts/01_download.py
python experiments/00_data_prep/scripts/02_build_manifest.py
python experiments/00_data_prep/scripts/03_split_subjects.py
cd experiments/03_loso_model_comparison
../../.venv/bin/python scripts/01_precompute.py
```

Smoke test:

```bash
../../.venv/bin/python scripts/06_loso_10model_queue.py audit
../../.venv/bin/python scripts/03_loso.py --model-id S9 --device cuda --epochs 1 --max-folds 1 --batch-size 8 --num-workers 0
```

Run on four GPUs:

```bash
cd experiments/03_loso_model_comparison
bash scripts/07_start_loso_10model_4gpu.sh
```

Generate the report:

```bash
../../.venv/bin/python scripts/06_loso_10model_queue.py report
```

Output:

```text
experiments/03_loso_model_comparison/result/loso_10model_compare_0p7m_<timestamp>/
```

Report package layout:

| File | Purpose |
|---|---|
| `REPORT.md` | Human-readable experiment report |
| `tables/model_summary.csv` | Ranked model-level summary |
| `tables/fold_results.csv` | Fold-level test metrics |
| `tables/model_architecture.csv` | Model structure and parameter table, if included in the release artifact |
| `reproducibility/environment.json` | Python / PyTorch / CUDA / GPU / git revision |
| `reproducibility/loso_10model_jobs_0p7m.yaml` | Exact job queue |
| `reproducibility/structure_compare_v2_0p7m.yaml` | Exact model/data/training config |

## Interpretation

This experiment should be described as a matched-capacity model-family comparison. Strict component ablation would require removing modules from one fixed full model; here each complete model is tuned to a similar parameter budget and compared under the same LOSO protocol.
