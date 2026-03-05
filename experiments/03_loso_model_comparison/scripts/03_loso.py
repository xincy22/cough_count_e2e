"""
LOSO评估脚本 - 对三个模型进行Leave-One-Subject-Out交叉验证
从experiment.yaml读取配置，为每个模型运行LOSO评估
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from coughcount.evaluation.edgeai import evaluate_run_on_split
from coughcount.losses import count_mae, sample_count_abs_error, train_loss_weighted
from coughcount.paths import ProjectPaths as P
from coughcount.training.edgeai import (
    build_dynamic_pos_neg_loss_balancer,
    evaluate_counting_metrics,
    prepare_training_components,
    save_epoch_artifacts,
    save_run_config,
)
from coughcount.training.loso import get_loso_splits
from coughcount.utils.io import atomic_write_json
from coughcount.utils.runtime import pick_device, set_seed


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _loss_balancer_state(
    loss_balancer: Any,
) -> dict[str, float | None]:
    return {
        "ema_pos_error": (
            float(loss_balancer.ema_pos_error)
            if loss_balancer.ema_pos_error is not None
            else None
        ),
        "ema_neg_error": (
            float(loss_balancer.ema_neg_error)
            if loss_balancer.ema_neg_error is not None
            else None
        ),
    }


def _restore_loss_balancer_state(loss_balancer: Any, state: dict[str, Any]) -> None:
    loss_balancer.ema_pos_error = (
        float(state["ema_pos_error"]) if state.get("ema_pos_error") is not None else None
    )
    loss_balancer.ema_neg_error = (
        float(state["ema_neg_error"]) if state.get("ema_neg_error") is not None else None
    )


def train_loso_fold(
    model_cfg: dict,
    cfg: dict[str, Any],
    train_split: list[str],
    val_split: list[str],
    test_subject: str,
    fold_idx: int,
    *,
    loso_run_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """训练单个LOSO fold"""

    loso_run_dir = Path(loso_run_dir)
    fold_dir = loso_run_dir / f"fold_{fold_idx:02d}_test_{test_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Prepare config for this fold
    fold_cfg = copy.deepcopy(cfg)
    fold_cfg["model"] = {
        "name": model_cfg["architecture"]["type"],
        "presets": {
            model_cfg["architecture"]["type"]: model_cfg["architecture"]
        }
    }

    # Set custom splits
    fold_cfg["data"]["custom_train_subjects"] = train_split
    fold_cfg["data"]["custom_val_subjects"] = val_split
    fold_cfg["data"]["custom_test_subjects"] = [test_subject]

    save_run_config(fold_dir, fold_cfg)

    components = prepare_training_components(fold_cfg, device=device)
    loss_balancer = build_dynamic_pos_neg_loss_balancer(fold_cfg)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_count = float("inf")

    epochs = components.epochs

    for epoch in range(1, epochs + 1):
        components.model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []

        pbar = tqdm(
            components.train_loader,
            desc=f"fold_{fold_idx:02d} e{epoch}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)

            sample_weights, is_pos, pos_ratio = loss_balancer.build_sample_weights(
                y,
                lengths,
                epoch=epoch,
            )

            pred = components.model(x)
            loss = train_loss_weighted(
                pred,
                y,
                lengths,
                count_loss_weight=components.count_loss_weight,
                sample_weights=sample_weights if loss_balancer.enabled else None,
            )

            components.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            components.optimizer.step()

            train_losses.append(float(loss.item()))
            train_maes.append(float(count_mae(pred, y, lengths).item()))
            with np.errstate(invalid="ignore"):
                batch_count_errors = sample_count_abs_error(pred.detach(), y, lengths)
            loss_balancer.update_from_batch_errors(batch_count_errors, is_pos)

            epoch_progress = (epoch - 1) + (pbar.n / max(1, len(components.train_loader)))
            components.scheduler.step(epoch_progress)

            lr_now = float(components.optimizer.param_groups[0]["lr"])
            pbar.set_postfix(
                mse=f"{np.mean(train_losses):.4f}",
                cmae=f"{np.mean(train_maes):.3f}",
                lr=f"{lr_now:.2e}",
            )

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        train_cmae = float(np.mean(train_maes)) if train_maes else float("nan")

        val_stats = evaluate_counting_metrics(
            components.model,
            components.val_loader,
            device,
            pos_threshold=components.pos_threshold,
            desc="val",
            disable=True,
        )
        val_mse = float(val_stats["mse"])
        val_cmae = float(val_stats["count_mae"])

        rec = {
            "epoch": int(epoch),
            "lr": lr_now,
            "train_mse": train_mse,
            "train_count_mae": train_cmae,
            "val_mse": val_mse,
            "val_count_mae": val_cmae,
            "val_count_mae_pos": float(val_stats["count_mae_pos"]),
            "val_count_mae_neg": float(val_stats["count_mae_neg"]),
        }
        history.append(rec)

        # Save best checkpoint
        if val_cmae < best_count:
            best_count = val_cmae
            best_val = val_mse
            ckpt = {
                "epoch": epoch,
                "model_state": components.model.state_dict(),
                "opt_state": components.optimizer.state_dict(),
                "scheduler_state": components.scheduler.state_dict(),
                "cfg": fold_cfg,
                "loss_balancer_state": _loss_balancer_state(loss_balancer),
                "val_mse": val_mse,
                "val_count_mae": val_cmae,
            }
            torch.save(ckpt, fold_dir / "best.pt")

    # Save history
    atomic_write_json(fold_dir / "history.json", history)

    # Evaluate on test set (left-out subject)
    test_stats = evaluate_counting_metrics(
        components.model,
        components.test_loader,
        device,
        pos_threshold=components.pos_threshold,
        desc="test",
        disable=True,
    )

    # Save test results
    test_results = {
        "fold": int(fold_idx),
        "test_subject": str(test_subject),
        "train_subjects": train_split,
        "val_subjects": val_split,
        "mse": float(test_stats["mse"]),
        "count_mae": float(test_stats["count_mae"]),
        "count_mae_pos": float(test_stats["count_mae_pos"]),
        "count_mae_neg": float(test_stats["count_mae_neg"]),
        "mean_pred_count_on_neg": float(test_stats["mean_pred_count_on_neg"]),
        "mean_gt_count_on_pos": float(test_stats["mean_gt_count_on_pos"]),
        "num_test_samples": int(len(components.test_loader.dataset)),
    }
    atomic_write_json(fold_dir / "test_results.json", test_results)

    return test_results


def run_loso_for_model(
    model_cfg: dict,
    cfg: dict[str, Any],
    *,
    loso_output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """为单个模型运行完整的LOSO评估"""

    loso_output_dir = Path(loso_output_dir)
    loso_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running LOSO for: {model_cfg['name']}")
    print(f"{'='*60}")

    # Get LOSO splits
    splits_path = Path(cfg["data"]["splits_json"])
    loso_splits = get_loso_splits(splits_path)

    all_results = []

    for fold_idx, (train_subjs, val_subjs, test_subj) in enumerate(loso_splits):
        print(f"\nFold {fold_idx + 1}/{len(loso_splits)}: Test subject {test_subj}")

        result = train_loso_fold(
            model_cfg=model_cfg,
            cfg=cfg,
            train_split=train_subjs,
            val_split=val_subjs,
            test_subject=test_subj,
            fold_idx=fold_idx,
            loso_run_dir=loso_output_dir,
            device=device,
        )

        all_results.append(result)
        print(f"  Test MAE: {result['count_mae']:.4f} (pos: {result['count_mae_pos']:.4f})")

    # Compute aggregate statistics
    count_maes = [r["count_mae"] for r in all_results]
    count_mae_poses = [r["count_mae_pos"] for r in all_results]
    count_mae_negs = [r["count_mae_neg"] for r in all_results]

    summary = {
        "model_id": model_cfg["id"],
        "model_name": model_cfg["name"],
        "num_folds": len(all_results),
        "mean_count_mae": float(np.mean(count_maes)),
        "std_count_mae": float(np.std(count_maes)),
        "mean_count_mae_pos": float(np.mean(count_mae_poses)),
        "std_count_mae_pos": float(np.std(count_mae_poses)),
        "mean_count_mae_neg": float(np.mean(count_mae_negs)),
        "std_count_mae_neg": float(np.std(count_mae_negs)),
        "fold_results": all_results,
    }

    # Save summary
    atomic_write_json(loso_output_dir / "loso_summary.json", summary)

    print(f"\n{model_cfg['name']} LOSO Summary:")
    print(f"  Mean Count MAE: {summary['mean_count_mae']:.4f} ± {summary['std_count_mae']:.4f}")
    print(f"  Mean Count MAE (pos): {summary['mean_count_mae_pos']:.4f} ± {summary['std_count_mae_pos']:.4f}")

    return summary


def main() -> None:
    import argparse

    cfg = load_config()

    parser = argparse.ArgumentParser(description="Run LOSO evaluation for model comparison.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Run LOSO only for a specific model (e.g., 'M1', 'M2', 'M3')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override",
    )
    args = parser.parse_args()

    # Get paths
    exp_dir = Path(__file__).parent.parent
    data_dir = exp_dir / "data"

    if not data_dir.exists():
        print(f"Error: data directory {data_dir} does not exist")
        print("Please run scripts/01_precompute.py first")
        return

    # Update config with data paths
    cfg["data"]["npy_dir"] = str(data_dir.resolve())
    cfg["data"]["splits_json"] = str((data_dir / "splits.json").resolve())

    if args.device is not None:
        cfg["training"]["device"] = str(args.device)

    device = cfg["training"]["device"]

    # Get output directory
    loso_cfg = cfg.get("loso", {})
    exp_name = loso_cfg.get("exp_name", "loso_model_compare")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loso_runs_dir = exp_dir / "runs" / f"{exp_name}_{timestamp}"
    loso_runs_dir.mkdir(parents=True, exist_ok=True)

    models = cfg["models"]
    if args.model_id:
        models = [m for m in models if m["id"] == args.model_id]
        if not models:
            print(f"Error: model_id '{args.model_id}' not found")
            return

    all_summaries = []

    for model_cfg in models:
        model_id = model_cfg["id"]
        model_output_dir = loso_runs_dir / model_id

        summary = run_loso_for_model(
            model_cfg=model_cfg,
            cfg=cfg,
            loso_output_dir=model_output_dir,
            device=device,
        )

        all_summaries.append(summary)

    # Save overall summary
    overall_summary = {
        "experiment": cfg["name"],
        "timestamp": timestamp,
        "device": device,
        "models_tested": len(all_summaries),
        "results": all_summaries,
    }
    atomic_write_json(loso_runs_dir / "loso_comparison_summary.json", overall_summary)

    # Print comparison table
    print(f"\n{'='*80}")
    print("LOSO MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Mean MAE':<12} {'Std MAE':<12} {'Mean Pos MAE':<14} {'Std Pos MAE':<14}")
    print("-" * 80)
    for s in all_summaries:
        print(
            f"{s['model_name']:<20} "
            f"{s['mean_count_mae']:<12.4f} "
            f"{s['std_count_mae']:<12.4f} "
            f"{s['mean_count_mae_pos']:<14.4f} "
            f"{s['std_count_mae_pos']:<14.4f}"
        )

    print(f"\nOverall summary saved to {loso_runs_dir / 'loso_comparison_summary.json'}")


if __name__ == "__main__":
    main()
