from __future__ import annotations

import argparse
import json
from pathlib import Path

from coughcount.evaluation.edgeai import evaluate_run_on_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained model on a data split.")
    ap.add_argument(
        "run_dir",
        type=Path,
        help="Path to training run directory containing best.pt or last.pt",
    )
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

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
