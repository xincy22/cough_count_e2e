from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(
    history_path: Path,
    *,
    output_path: Path | None = None,
    show: bool = False,
) -> Path:
    history_path = Path(history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"File not found: {history_path}")

    with history_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("History file is empty.")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    run_name = history_path.parent.name
    fig.suptitle(f"Training History: {run_name}", fontsize=16)

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_mse"], label="Train MSE")
    if "val_mse" in df.columns:
        ax.plot(df["epoch"], df["val_mse"], label="Val MSE")
    ax.set_title("MSE Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_count_mae"], label="Train MAE")
    if "val_count_mae" in df.columns:
        ax.plot(df["epoch"], df["val_count_mae"], label="Val MAE")
    ax.set_title("Count MAE (Mean Absolute Error)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    has_breakdown = False
    if "val_count_mae_pos" in df.columns:
        ax.plot(
            df["epoch"],
            df["val_count_mae_pos"],
            label="Val MAE (Positive)",
            linestyle="--",
        )
        has_breakdown = True
    if "val_count_mae_neg" in df.columns:
        ax.plot(
            df["epoch"],
            df["val_count_mae_neg"],
            label="Val MAE (Negative)",
            linestyle="--",
        )
        has_breakdown = True
    if "val_count_mae" in df.columns and has_breakdown:
        ax.plot(
            df["epoch"],
            df["val_count_mae"],
            label="Val MAE (Overall)",
            color="black",
            alpha=0.5,
        )
    if has_breakdown:
        ax.set_title("Validation MAE Breakdown")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No breakdown data available", ha="center", va="center")

    ax = axes[1, 1]
    if "lr" in df.columns:
        ax.plot(df["epoch"], df["lr"], color="green")
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
        if df["lr"].max() / (df["lr"].min() + 1e-9) > 100:
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No LR data available", ha="center", va="center")

    plt.tight_layout()

    out_path = Path(output_path) if output_path is not None else history_path.parent / "training_curves.png"
    plt.savefig(out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path
