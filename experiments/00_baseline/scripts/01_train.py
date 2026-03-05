"""
Baseline模型训练脚本
使用最佳density核配置 (skewed_20_120) 训练1000 epochs
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

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


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _loss_balancer_state(loss_balancer) -> dict:
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


def main() -> None:
    cfg = load_config()
    exp_dir = Path(__file__).parent.parent

    # 使用02实验的预计算数据
    data_dir = exp_dir.parent / "02_density_kernel" / "data" / cfg["density"]["data_dir"]

    if not data_dir.exists():
        print(f"Error: 数据目录不存在: {data_dir}")
        print("请先运行 02_density_kernel 的 01_precompute.py")
        return

    print("="*60)
    print("BASELINE模型训练")
    print("="*60)
    print(f"配置: {cfg['description']}")
    print(f"数据: {data_dir}")
    print()

    # 准备配置
    train_cfg = copy.deepcopy(cfg)
    train_cfg["data"]["npy_dir"] = str(data_dir.resolve())
    train_cfg["data"]["splits_json"] = str((data_dir / "splits.json").resolve())

    # 创建运行目录
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(run_dir, train_cfg)

    # 准备训练组件
    from coughcount.utils.runtime import pick_device, set_seed
    set_seed(cfg["seed"])
    device = pick_device(cfg["training"].get("device", "cuda"))

    components = prepare_training_components(train_cfg, device=device)
    loss_balancer = build_dynamic_pos_neg_loss_balancer(train_cfg)

    print(f"模型: {cfg['model']['name']}")
    print(f"参数量: {count_trainable_params(components.model):,}")
    print(f"训练样本: {len(components.train_dataset)} (pos={len(components.train_dataset.pos_idx)}, neg={len(components.train_dataset.neg_idx)})")
    print(f"验证样本: {len(components.val_dataset)} (pos={components.val_pos_windows}, neg={components.val_neg_windows})")
    print()

    # 训练循环
    history: list[dict] = []
    best_val = float("inf")
    best_count = float("inf")

    epochs = components.epochs

    for epoch in range(1, epochs + 1):
        components.model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []

        pbar = tqdm(
            components.train_loader,
            desc=f"epoch {epoch}/{epochs}",
            dynamic_ncols=True,
        )
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)

            sample_weights, is_pos, pos_ratio = loss_balancer.build_sample_weights(
                y, lengths, epoch=epoch
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
                pos_a=f"{pos_ratio:.2f}" if loss_balancer.enabled else "0.50",
            )

        train_mse = float(np.mean(train_losses))
        train_cmae = float(np.mean(train_maes))

        val_stats = evaluate_counting_metrics(
            components.model,
            components.val_loader,
            device,
            pos_threshold=components.pos_threshold,
            desc="val",
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

        print(
            f"[epoch {epoch}] "
            f"train_mse={train_mse:.6f} train_cmae={train_cmae:.4f} "
            f"val_mse={val_mse:.6f} val_cmae={val_cmae:.4f} "
            f"(pos={val_stats['count_mae_pos']:.3f} neg={val_stats['count_mae_neg']:.3f})"
        )

        best_val, best_count = save_epoch_artifacts(
            run_dir=run_dir,
            cfg=train_cfg,
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

    # 保存训练汇总
    out = {
        "best_val_mse": float(best_val),
        "best_val_count_mae": float(best_count),
        "epochs_total": int(epochs),
        "run_dir": str(run_dir.resolve()),
    }
    atomic_write_json(run_dir / "train_summary.json", out)

    print()
    print("="*60)
    print("训练完成!")
    print(f"Best Val MSE: {best_val:.6f}")
    print(f"Best Val Count MAE: {best_count:.4f}")
    print("="*60)

    # 在测试集上评估
    print()
    print("在测试集上评估...")

    test_metrics, out_file, ckpt_path = evaluate_run_on_split(
        run_dir,
        split="test",
        batch_size=cfg["loader"]["batch_size"],
        num_workers=cfg["loader"]["num_workers"],
        device_name=str(device),
    )

    print()
    print("="*60)
    print("BASELINE结果")
    print("="*60)
    print(f"Test Count MAE: {test_metrics['count_mae']:.4f}")
    print(f"Test Count MAE (pos): {test_metrics['count_mae_pos']:.4f}")
    print(f"Test Count MAE (neg): {test_metrics['count_mae_neg']:.4f}")
    print()

    # 检查是否达到目标
    targets = cfg.get("targets", {})
    if targets:
        print("目标对比:")
        for key, target in targets.items():
            actual = test_metrics.get(key, float("inf"))
            status = "✓" if actual <= target else "✗"
            print(f"  {key}: {actual:.4f} (目标: {target:.4f}) {status}")

    print()
    print(f"模型保存在: {run_dir}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
