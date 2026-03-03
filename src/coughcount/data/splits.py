from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def list_subject_ids(manifest_csv: Path) -> list[str]:
    df = pd.read_csv(manifest_csv)
    return sorted(df["subject_id"].astype(str).unique().tolist())


def _write_splits_json(splits: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def make_loso_subject_splits(
    manifest_csv: Path,
    out_dir: Path,
    *,
    val_subjects: int = 1,
    seed: int = 42,
) -> list[dict[str, object]]:
    """
    Build leave-one-subject-out split files.

    For each subject s:
      - test = [s]
      - val = sampled from remaining subjects
      - train = remaining - val
    """
    subjects = list_subject_ids(manifest_csv)
    n = len(subjects)
    if n < 3:
        raise ValueError("Need at least 3 subjects for LOSO with val split.")

    val_subjects = int(val_subjects)
    if val_subjects < 1 or val_subjects >= (n - 1):
        raise ValueError(
            f"val_subjects must be in [1, {n - 2}], got {val_subjects}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict[str, object]] = []
    for fold_idx, test_subject in enumerate(subjects):
        candidates = [s for s in subjects if s != test_subject]
        rng = np.random.default_rng(seed + fold_idx)
        val = sorted(rng.choice(candidates, size=val_subjects, replace=False).tolist())
        val_set = set(val)
        train = [s for s in candidates if s not in val_set]

        split_obj: dict[str, object] = {
            "train": train,
            "val": val,
            "test": [test_subject],
            "meta": {
                "mode": "loso",
                "fold_index": fold_idx,
                "seed": seed,
                "val_subjects": val_subjects,
                "n_total": n,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": 1,
                "test_subject": test_subject,
            },
        }

        split_path = out_dir / f"fold_{fold_idx:02d}_{test_subject}.json"
        _write_splits_json(split_obj, split_path)

        folds.append(
            {
                "fold_index": fold_idx,
                "test_subject": test_subject,
                "val_subjects": val,
                "train_subjects": train,
                "splits_json": str(split_path),
            }
        )

    index_obj: dict[str, object] = {
        "mode": "loso",
        "manifest_csv": str(Path(manifest_csv)),
        "seed": seed,
        "val_subjects": val_subjects,
        "n_subjects": n,
        "n_folds": len(folds),
        "folds": folds,
    }
    _write_splits_json(index_obj, out_dir / "index.json")
    return folds


def make_holdout_subject_split(
    manifest_csv: Path,
    out_path: Path,
    *,
    val_subjects: int = 2,
    test_subjects: int = 2,
    seed: int = 42,
) -> dict[str, object]:
    """
    Build a single subject-level holdout split:
      - test = random subjects
      - val = random subjects from remaining
      - train = rest
    """
    subjects = list_subject_ids(manifest_csv)
    n = len(subjects)
    val_subjects = int(val_subjects)
    test_subjects = int(test_subjects)
    if n < (val_subjects + test_subjects + 1):
        raise ValueError(
            "Not enough subjects for holdout split. "
            f"Need at least {val_subjects + test_subjects + 1}, got {n}."
        )

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(subjects).tolist()

    test = sorted(perm[:test_subjects])
    val = sorted(perm[test_subjects : test_subjects + val_subjects])
    train = sorted(perm[test_subjects + val_subjects :])

    split_obj: dict[str, object] = {
        "train": train,
        "val": val,
        "test": test,
        "meta": {
            "mode": "holdout",
            "seed": int(seed),
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "n_total": n,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
    }

    _write_splits_json(split_obj, Path(out_path))
    return split_obj
