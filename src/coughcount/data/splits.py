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
