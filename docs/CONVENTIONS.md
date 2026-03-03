# Conventions

This repo follows a simple rule: keep **experiments self-contained**, and keep **results curated**.

## Folder Contract

### `scripts/`

Shared, one-off pipeline steps that are not tied to a specific experiment, for example:

- download dataset
- build manifest
- precompute windows/features
- live mic inference

### `experiments/<exp_name>/`

Everything needed to **run** one experiment lives here.

Required:

- `experiment.yaml`: the single config entrypoint for the experiment
- `scripts/`: numbered scripts to run in order (`01_*.py`, `02_*.py`, ...)

Generated (ignored by git):

- `runs/`: raw training outputs (checkpoints, histories, per-fold dirs, etc.)
- `splits/`: holdout/LOSO split files used by the experiment

Optional:

- `configs/`: experiment-local model/base configs (to avoid cross-experiment coupling)

### `results/<exp_name>/`

Everything that is **curated and presentable** for the experiment lives here.

Keep this small and human-facing:

- `NOTES.md` / `REPORT.md`
- `tables/` (CSV)
- `figures/` (PNG)
- `packages/` (small zip + manifest, optional)

Avoid putting raw training outputs here.

## Naming Rules

- Use `NN_<short_name>` for experiments, for example:
  - `01_loso_model_compare`
  - `02_tcn_gru_structure`
- `results/<exp_name>/` must have the **same `<exp_name>`** as `experiments/<exp_name>/`.

## CLI / Config Rules

- Prefer “one script does one thing”.
- Scripts should read `experiment.yaml` by default, and avoid complex `argparse`.
- Resume behavior should be safe-by-default (continue from `last.pt` when present, skip finished folds when possible).

## Git / Release Rules

- Commit: `experiments/<exp>/scripts/`, `experiment.yaml`, and `results/<exp>/` (curated artifacts).
- Do not commit: `experiments/<exp>/runs/`, `experiments/<exp>/splits/`, `data/`, `runs/`.
- Model weights (`*.pt`) are ignored by default; publish them via GitHub Releases if needed.

## Why This Scales

- Adding a new experiment is trivial: create `experiments/<exp>/` + `results/<exp>/`.
- Experiments do not step on each other’s runs/splits.
- Results remain readable, reviewable, and suitable for open-source releases.

