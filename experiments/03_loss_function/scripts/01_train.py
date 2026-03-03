"""
训练脚本 - 为所有损失函数变体训练模型
从experiment.yaml读取配置，迭代训练所有6种损失函数配置
"""
from __future__ import annotations

import copy
import json
import statistics
import time
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


def _infer_best_values(history: list[dict[str, Any]]) -> tuple[float, float]:
    if not history:
        return float("inf"), float("inf")
    best_val = float(min(float(h.get("val_mse", float("inf"))) for h in history))
    best_count = float(
        min(float(h.get("val_count_mae", float("inf"))) for h in history)
    )
    return best_val, best_count


def train_single_run(
    cfg: dict[str, Any],
    *,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir, cfg)

    train_cfg = cfg.get("training", {})
    device = pick_device(train_cfg.get("device", "cuda"))
    print(f"\n[train] run_dir={run_dir}")
    print(f"[train] device={device}")

    components = prepare_training_components(cfg, device=device)
    loss_balancer = build_dynamic_pos_neg_loss_balancer(cfg)

    print(
        f"[train] train_windows={len(components.train_dataset)} "
        f"(pos={len(components.train_dataset.pos_idx)} neg={len(components.train_dataset.neg_idx)})"
    )
    print(
        f"[train] val_windows={len(components.val_dataset)} "
        f"(pos={components.val_pos_windows} neg={components.val_neg_windows})"
        f"{' [balanced 1:1]' if components.val_is_balanced else ''}"
    )
    print(
        f"[train] model={cfg['model']['name']} "
        f"params={count_trainable_params(components.model):,}"
    )

    history_path = run_dir / "history.json"
    ckpt_last = run_dir / "last.pt"

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_count = float("inf")

    if history_path.exists():
        loaded = _load_json(history_path)
        if isinstance(loaded, list):
            history = loaded
            best_val, best_count = _infer_best_values(history)

    if resume and ckpt_last.exists():
        ckpt = torch.load(ckpt_last, map_location="cpu")
        components.model.load_state_dict(ckpt["model_state"])
        components.optimizer.load_state_dict(ckpt["opt_state"])
        if ckpt.get("scheduler_state") is not None:
            components.scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("loss_balancer_state") is not None:
            _restore_loss_balancer_state(loss_balancer, ckpt["loss_balancer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[train] resume from epoch {start_epoch}")

    for epoch in range(start_epoch, components.epochs + 1):
        components.model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []

        pbar = tqdm(
            components.train_loader,
            desc=f"train e{epoch}/{components.epochs}",
            dynamic_ncols=True,
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
                under_count_weight=cfg["training"].get("under_count_weight", 0.0),
                under_count_positive_only=cfg["training"].get("under_count_positive_only", True),
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
                pos_a=f"{pos_ratio:.2f}" if loss_balancer.enabled else "0.50",
            )

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        train_cmae = float(np.mean(train_maes)) if train_maes else float("nan")

        val_stats = evaluate_counting_metrics(
            components.model,
            components.val_loader,
            device,
            pos_threshold=components.pos_threshold,
            desc="val",
        )
        val_mse = float(val_stats["mse"])
        val_cmae = float(val_stats["count_mae"])
        lr_now = float(components.optimizer.param_groups[0]["lr"])

        rec = {
            "epoch": int(epoch),
            "lr": lr_now,
            "train_mse": train_mse,
            "train_count_mae": train_cmae,
            "val_mse": val_mse,
            "val_count_mae": val_cmae,
            "val_count_mae_pos": float(val_stats["count_mae_pos"]),
            "val_count_mae_neg": float(val_stats["count_mae_neg"]),
            "val_mean_pred_count_on_neg": float(val_stats["mean_pred_count_on_neg"]),
            "val_mean_gt_count_on_pos": float(val_stats["mean_gt_count_on_pos"]),
        }
        history.append(rec)

        print(
            f"[epoch {epoch}] "
            f"lr={lr_now:.2e} "
            f"train_mse={train_mse:.6f} train_count_mae={train_cmae:.4f} "
            f"val_mse={val_mse:.6f} val_count_mae={val_cmae:.4f} "
            f"(pos_mae={val_stats['count_mae_pos']:.3f} neg_mae={val_stats['count_mae_neg']:.3f})"
        )

        best_val, best_count = save_epoch_artifacts(
            run_dir=run_dir,
            cfg=cfg,
            epoch=epoch,
            model=components.model,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
            val_mse=val_mse,
            val_count_mae=val_cmae,
            best_val_mse=best_val,
            best_val_count_mae=best_count,
            count_loss_weight=components.count_loss_weight,
            epoch_metrics=rec,
            history=history,
            extra_state={"loss_balancer_state": _loss_balancer_state(loss_balancer)},
        )

    out = {
        "best_val_mse": float(best_val),
        "best_val_count_mae": float(best_count),
        "epochs_total": int(components.epochs),
        "epochs_ran": int(max(0, components.epochs - start_epoch + 1)),
        "run_dir": str(run_dir),
    }
    atomic_write_json(run_dir / "train_summary.json", out)
    return out


def main() -> None:
    import argparse

    cfg = load_config()

    # Parse CLI args for runtime overrides
    parser = argparse.ArgumentParser(description="Train all loss function variants.")
    parser.add_argument(
        "--loss-id",
        type=str,
        default=None,
        help="Train only a specific loss config (e.g., '2A', '2B', etc.)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs override",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing checkpoints",
    )
    args = parser.parse_args()

    resume = not bool(args.no_resume)

    # Get output directory
    exp_dir = Path(__file__).parent.parent
    runs_dir = Path(cfg["output"]["runs_dir"])
    if not runs_dir.is_absolute():
        runs_dir = exp_dir / runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    loss_configs = cfg["loss_configs"]
    if args.loss_id:
        loss_configs = [lc for lc in loss_configs if lc["id"] == args.loss_id]
        if not loss_configs:
            print(f"Error: loss_id '{args.loss_id}' not found")
            return

    results = []

    for loss_cfg in loss_configs:
        loss_id = loss_cfg["id"]
        loss_name = loss_cfg["name"]

        # Prepare config for this loss variant
        loss_run_cfg = copy.deepcopy(cfg)
        loss_run_cfg["training"]["count_loss_weight"] = loss_cfg["count_loss_weight"]
        loss_run_cfg["training"]["under_count_weight"] = loss_cfg.get("under_count_weight", 0.0)
        loss_run_cfg["training"]["under_count_positive_only"] = loss_cfg.get("under_count_positive_only", True)

        if args.epochs is not None:
            loss_run_cfg["training"]["epochs"] = int(args.epochs)
        if args.device is not None:
            loss_run_cfg["training"]["device"] = str(args.device)

        run_dir = runs_dir / loss_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Loss Config {loss_id}: {loss_name}")
        print(f"{'='*60}")
        print(f"count_loss_weight: {loss_cfg['count_loss_weight']}")
        print(f"under_count_weight: {loss_cfg.get('under_count_weight', 0.0)}")
        print(f"Run dir: {run_dir}")
        print(f"Description: {loss_cfg['description']}")

        train_summary = train_single_run(loss_run_cfg, run_dir=run_dir, resume=resume)

        # Evaluate on test set
        eval_batch = int(loss_run_cfg.get("loader", {}).get("batch_size", 16))
        eval_workers = int(loss_run_cfg.get("loader", {}).get("num_workers", 4))
        eval_device = str(loss_run_cfg.get("training", {}).get("device", "cuda"))
        test_metrics, out_file, ckpt_path = evaluate_run_on_split(
            run_dir,
            split="test",
            batch_size=eval_batch,
            num_workers=eval_workers,
            device_name=eval_device,
        )

        result = {
            "loss_id": loss_id,
            "loss_name": loss_name,
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(ckpt_path.resolve()),
            "loss_config": loss_cfg,
            "train_summary": train_summary,
            "test_metrics": test_metrics,
        }
        results.append(result)

        print(f"\n[{loss_id}] Test results:")
        print(f"  count_mae: {test_metrics['count_mae']:.4f}")
        print(f"  count_mae_pos: {test_metrics['count_mae_pos']:.4f}")
        print(f"  count_mae_neg: {test_metrics['count_mae_neg']:.4f}")

    # Save summary
    summary = {
        "experiment": cfg["name"],
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "loss_configs_tested": len(results),
        "results": results,
    }
    summary_path = runs_dir / "loss_comparison_summary.json"
    atomic_write_json(summary_path, summary)

    # Print comparison table
    print(f"\n{'='*80}")
    print("LOSS FUNCTION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'ID':<5} {'Name':<25} {'Val MAE':<10} {'Val Pos MAE':<12} {'Test MAE':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['loss_id']:<5} "
            f"{r['loss_name']:<25} "
            f"{r['train_summary']['best_val_count_mae']:<10.4f} "
            f"{r['test_metrics']['count_mae_pos']:<12.4f} "
            f"{r['test_metrics']['count_mae']:<10.4f}"
        )

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
