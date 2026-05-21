# Chapter 5 0.7M 10-Model Remote Guide

This is the remote-running guide for the final Chapter 5 experiment. The workflow is a single 10-model LOSO comparison, not a separate ablation experiment.

## Remote Setup

Run from the repository root on the remote machine:

```bash
uv sync --locked
```

If `uv` is unavailable in the rental image:

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
```

Verify PyTorch and GPUs:

```bash
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected precomputed data location:

```text
experiments/03_loso_model_comparison/data/
```

If the directory is missing, either copy the precomputed package or run:

```bash
cd experiments/03_loso_model_comparison
../../.venv/bin/python scripts/01_precompute.py
```

## What Runs

The final queue contains 10 LOSO jobs:

```text
S0 CNN1D
S1 DSCNN
S2 ResCNN
S3 CRNN
S4 BiCRNN
S5 BiGRU
S6 TCN
S7 TCN-Attn
S8 TCN+UniGRU
S9 TCN+BiGRU
```

All models use the same LOSO protocol and are kept near a 0.7M trainable-parameter budget.

## 4-GPU Run

```bash
cd experiments/03_loso_model_comparison
bash scripts/07_start_ch5_rerun_4gpu.sh
```

Equivalent manual commands:

```bash
cd experiments/03_loso_model_comparison
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0 > shard0.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 1 --num-shards 4 --device cuda:1 > shard1.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 2 --num-shards 4 --device cuda:2 > shard2.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 3 --num-shards 4 --device cuda:3 > shard3.out 2>&1 &
```

With 10 jobs and 4 GPUs, shards 0 and 1 run 3 jobs each, and shards 2 and 3 run 2 jobs each.

## Before Running

Audit exact parameters and model IDs:

```bash
../../.venv/bin/python scripts/06_ch5_rerun_queue.py audit
```

Dry-run one shard:

```bash
../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0 --dry-run
```

## During Running

```bash
nvidia-smi
ps -ef | grep 06_ch5_rerun_queue.py | grep -v grep
tail -f shard0.out
tail -f runs/ch5_10model_compare_0p7m_logs/J00_cnn1d_0p7m.log
```

## After All Shards Finish

Generate the release-ready report:

```bash
../../.venv/bin/python scripts/06_ch5_rerun_queue.py report
```

Output:

```text
result/ch5_10model_compare_0p7m_<timestamp>/
```

Important files:

```text
RELEASE_REPORT.md
model_comparison_summary.csv
fold_results_all.csv
missing_jobs.csv
environment.json
ch5_rerun_jobs_0p7m.yaml
structure_compare_v2_0p7m.yaml
```
