from __future__ import annotations

import json
from pathlib import Path

import yaml

from coughcount.evaluation.edgeai import evaluate_run_on_split


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> None:
    import argparse

    cfg = load_config()

    # Parse CLI args for run_dir (required), use config defaults for optional args
    parser = argparse.ArgumentParser(description="Evaluate trained model on a data split.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to training run directory containing best.pt or last.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=cfg.get("test", {}).get("split", "test"),
        help="Data split to evaluate on (default: from config or 'test')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.get("test", {}).get("batch_size", cfg.get("loader", {}).get("batch_size", 32)),
        help="Batch size for evaluation (default: from config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=cfg.get("test", {}).get("num_workers", cfg.get("loader", {}).get("num_workers", 4)),
        help="Number of data loader workers (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.get("training", {}).get("device", "cuda"),
        help="Device to use (default: from config)",
    )
    args = parser.parse_args()

    metrics, out_file, ckpt_path = evaluate_run_on_split(
        args.run_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Split: {args.split}")
    print("\nEvaluation Results:")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
