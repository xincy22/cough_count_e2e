from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def make_subject_splits(
    manifest_csv: Path,
    splits_json: Path,
    *,
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[str]]:

    if train_ratio < 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("Invalid train/val ratios")

    df = pd.read_csv(manifest_csv)
    subjects = sorted(df["subject_id"].astype(str).unique().tolist())

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(1, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    n_test = n - n_train - n_val

    train = subjects[:n_train]
    val = subjects[n_train : n_train + n_val]
    test = subjects[n_train + n_val :]

    splits_json.parent.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train,
        "val": val,
        "test": test,
        "meta": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "n_total": n,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        },
    }
    with splits_json.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return splits


def load_subject_ids(manifest_csv: Path) -> list[str]:
    df = pd.read_csv(manifest_csv)
    return sorted(df["subject_id"].astype(str).unique().tolist())


def build_loso_fold(
    subjects: list[str],
    *,
    fold_index: int,
) -> dict[str, list[str] | dict[str, int]]:
    if len(subjects) < 3:
        raise ValueError("LOSO requires at least 3 subjects to build train/val/test.")

    n = len(subjects)
    i = int(fold_index) % n
    test_subject = subjects[i]
    val_subject = subjects[(i + 1) % n]
    train_subjects = [s for s in subjects if s not in {test_subject, val_subject}]

    return {
        "train": train_subjects,
        "val": [val_subject],
        "test": [test_subject],
        "meta": {
            "protocol": "loso_with_subject_val",
            "fold_index": int(i),
            "n_total_subjects": int(n),
        },
    }


def write_subject_splits(
    splits_json: Path,
    *,
    train: list[str],
    val: list[str],
    test: list[str],
    meta: dict[str, int | str | float] | None = None,
) -> dict[str, list[str] | dict[str, int | str | float]]:
    payload: dict[str, list[str] | dict[str, int | str | float]] = {
        "train": [str(x) for x in train],
        "val": [str(x) for x in val],
        "test": [str(x) for x in test],
        "meta": dict(meta or {}),
    }
    splits_json.parent.mkdir(parents=True, exist_ok=True)
    with splits_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
