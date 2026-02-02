import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys


def plot_history(history_path):
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"Error: File {history_path} not found.")
        sys.exit(1)

    # Load data
    with open(history_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if df.empty:
        print("Error: History file is empty.")
        return

    # Setup plots
    # We'll create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    run_name = history_path.parent.name
    fig.suptitle(f"Training History: {run_name}", fontsize=16)

    # 1. MSE Loss (Log scale usually helps if difference is large, but let's stick to linear first or try semilogy if numbers are small)
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_mse"], label="Train MSE")
    if "val_mse" in df.columns:
        ax.plot(df["epoch"], df["val_mse"], label="Val MSE")
    ax.set_title("MSE Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Count MAE
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_count_mae"], label="Train MAE")
    if "val_count_mae" in df.columns:
        ax.plot(df["epoch"], df["val_count_mae"], label="Val MAE")
    ax.set_title("Count MAE (Mean Absolute Error)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Val Breakdown (Positive vs Negative samples)
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

    # Also plot overall Val MAE again for comparison
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

    # 4. Learning Rate
    ax = axes[1, 1]
    if "lr" in df.columns:
        ax.plot(df["epoch"], df["lr"], color="green")
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
        # Use log scale for LR if it varies by orders of magnitude
        if df["lr"].max() / (df["lr"].min() + 1e-9) > 100:
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No LR data available", ha="center", va="center")

    plt.tight_layout()

    # Save plot
    save_path = history_path.parent / "training_curves.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show() # Comment out for headless environments or if running on remote


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize training history from JSON log."
    )
    parser.add_argument("history_file", type=str, help="Path to history.json file")
    args = parser.parse_args()

    plot_history(args.history_file)
