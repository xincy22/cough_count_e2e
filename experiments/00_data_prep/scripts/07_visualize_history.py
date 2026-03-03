from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from coughcount.viz.history import plot_training_history


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Visualize training history from JSON log."
    )
    parser.add_argument(
        "history_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to history.json file (optional, uses config default if not provided)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window in addition to saving it.",
    )
    args = parser.parse_args()

    # Use config default if no path provided
    history_file = args.history_file
    if history_file is None:
        default_path = cfg.get("visualization", {}).get("default_history_file")
        if default_path:
            history_file = Path(default_path)
        else:
            parser.error("history_file argument required (no default in config)")

    out_path = plot_training_history(history_file, show=args.show)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
