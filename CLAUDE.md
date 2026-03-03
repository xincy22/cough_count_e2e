# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end cough counting from audio using density map regression. The model predicts a frame-level density map from STFT features, then sums to get cough counts per audio window.

**Core approach**:
- Input: STFT log-magnitude features [F, T]
- Output: Frame-level density map [T], where sum(density) ≈ cough count
- Training: MSE on density maps + auxiliary count MAE loss

## Architecture

### Data Flow

```
audio.wav → STFT → S [F, T] → Model → density [T] → sum → count
                     ↓
                cough labels (start, end times)
                     ↓
                density kernel (gaussian/skewed/cosine)
                     ↓
                target density [T]
```

### Key Components

**Density Generation** (`src/coughcount/data/density.py`):
- Converts cough event timestamps to frame-level density maps
- Kernels: `gaussian` (symmetric), `skewed_gaussian` (left/right sigma), `cosine`
- `make_density(centers_sec, frame_times, kernel, **kwargs)` → (frame_times, density)

**Models** (`src/coughcount/models/`):
- `TCN`: Temporal Convolutional Network (baseline)
- `TCNGRU`: TCN + GRU layer
- `CRNN`: Conv1D + BiGRU
- `DSCNN`: Depthwise Separable Conv (budget1m variant)

**Training** (`src/coughcount/training/edgeai.py`, `loso.py`):
- Dynamic pos/neg loss balancer (adapts sample weighting based on errors)
- Multi-component loss: frame MSE + count MAE + under-count penalty
- LOSO (Leave-One-Subject-Out) evaluation for 15 subjects

### Data Structure

Precomputed data format:
```
<npy_dir>/
├── <sample_stem>/
│   ├── S.npy          # STFT features [F, T]
│   ├── t.npy          # frame times [T]
│   ├── density.npy    # target density [T]
│   └── meta.json      # metadata (subject_id, class, etc.)
```

## Experiment Conventions

### Directory Structure

**Self-contained experiments** (CRITICAL):
- Each experiment is **completely independent** - no code sharing between experiments
- All scripts needed live in `experiments/<exp_name>/scripts/`
- Run scripts in numbered order (01_*.py, 02_*.py, ...)
- **No cross-experiment imports or dependencies**

```
experiments/<exp_name>/
├── experiment.yaml          # ALL configuration here
├── scripts/                 # All code for this experiment
│   ├── 01_preprocess.py    # python scripts/01_preprocess.py
│   ├── 02_train.py         # python scripts/02_train.py
│   └── 03_evaluate.py      # python scripts/03_evaluate.py
├── data/                    # This experiment's data
├── runs/                    # Training outputs (gitignored)
└── README.md                # Experiment description
```

**Configuration rules**:
- All parameters in `experiment.yaml` (no argparse, no CLI args)
- Scripts read yaml: `config = yaml.safe_load(open("experiment.yaml"))`
- One config file = single source of truth

**Running an experiment**:
```bash
cd experiments/<exp_name>
python scripts/01_preprocess.py
python scripts/02_train.py
python scripts/03_evaluate.py
```

**No .sh scripts, no CLI arguments** - just numbered Python scripts.

**Numbered experiment prefixes**:
- `00_*`: Data prep / infrastructure
- `01_*`: Baseline models
- `02_*`: First optimization iteration
- `03_*`: Second optimization iteration, etc.

### Creating a New Experiment

```bash
mkdir -p experiments/03_my_experiment/{scripts,runs,data}

# Create experiment.yaml (ALL config, no CLI args)

# Create numbered scripts:
# scripts/01_preprocess.py  # config = yaml.safe_load(open("experiment.yaml"))
# scripts/02_train.py
# scripts/03_evaluate.py

# Run:
python scripts/01_preprocess.py
python scripts/02_train.py
python scripts/03_evaluate.py
```

**Key principles**:
- One `experiment.yaml` per experiment
- Scripts read yaml, not argparse
- Direct `python` commands, no shell wrappers
- All code in `scripts/` (copy from `src/` if needed)

### Data Locations

- **Raw data**: `data/raw/edgeai/public_dataset/` (downloaded once)
- **Processed data**: `experiments/<exp>/data/` (per-experiment)
- **Splits**: `data/processed/edgeai/splits.json` (shared or generated per-experiment)

## Dependencies

Install via `uv sync` or `python -m pip install -e .`

Key dependencies:
- PyTorch (CUDA 12.6)
- torchaudio, torchvision
- numpy, pandas, soundfile, scipy
- tqdm, pyyaml, matplotlib
