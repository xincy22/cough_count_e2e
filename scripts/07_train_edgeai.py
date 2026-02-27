from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from coughcount.losses import count_mae, sample_count_abs_error, train_loss_weighted
from coughcount.paths import ProjectPaths as P
from coughcount.training.edgeai import (
    build_dynamic_pos_neg_loss_balancer,
    create_run_dir,
    evaluate_counting_metrics,
    prepare_training_components,
    save_epoch_artifacts,
    save_run_config,
)
from coughcount.utils.config import load_yaml_config
from coughcount.utils.runtime import pick_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=P.configs / "edgeai.yaml")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg.get("seed", 0)))

    train_cfg = cfg.get("train", {})
    device = pick_device(train_cfg.get("device", "cuda"))
    print(f"Device: {device}")

    components = prepare_training_components(cfg, device=device)
    loss_balancer = build_dynamic_pos_neg_loss_balancer(cfg)

    run_dir = create_run_dir(cfg)
    save_run_config(run_dir, cfg)

    print(f"Run dir: {run_dir}")
    print(
        f"Train windows: {len(components.train_dataset)} "
        f"(pos={len(components.train_dataset.pos_idx)} neg={len(components.train_dataset.neg_idx)})"
    )
    print(
        f"Val windows:   {len(components.val_dataset)} "
        f"(pos={len(components.val_dataset.pos_idx)} neg={len(components.val_dataset.neg_idx)})"
    )
    print(
        f"Model: {cfg['model']['name']} preset={cfg['model']['presets'][cfg['model']['name']]}"
    )
    print(
        "LR schedule: "
        f"CosineAnnealingWarmRestarts(T_0={components.cycle_epochs}, eta_min={components.eta_min:g})"
    )
    if loss_balancer.enabled:
        print(
            "Dynamic loss: "
            f"enabled (alpha={loss_balancer.alpha:g}, "
            f"min_pos_weight={loss_balancer.min_pos_weight:g}, "
            f"max_pos_weight={loss_balancer.max_pos_weight:g}, "
            f"warmup_epochs={loss_balancer.warmup_epochs})"
        )

    best_val = float("inf")
    best_count = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, components.epochs + 1):
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

            sample_weights, is_pos, pos_weight = loss_balancer.build_sample_weights(
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

            epoch_progress = (epoch - 1) + (
                pbar.n / max(1, len(components.train_loader))
            )
            components.scheduler.step(epoch_progress)

            lr_now = float(components.optimizer.param_groups[0]["lr"])
            pbar.set_postfix(
                mse=f"{np.mean(train_losses):.4f}",
                cmae=f"{np.mean(train_maes):.3f}",
                lr=f"{lr_now:.2e}",
                pos_w=f"{pos_weight:.2f}" if loss_balancer.enabled else "1.00",
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
            f"(pos_mae={val_stats['count_mae_pos']:.3f} neg_mae={val_stats['count_mae_neg']:.3f} "
            f"neg_pred={val_stats['mean_pred_count_on_neg']:.2f})"
        )

        best_val, best_count = save_epoch_artifacts(
            run_dir=run_dir,
            cfg=cfg,
            epoch=epoch,
            model=components.model,
            optimizer=components.optimizer,
            val_mse=val_mse,
            val_count_mae=val_cmae,
            best_val_mse=best_val,
            best_val_count_mae=best_count,
            count_loss_weight=components.count_loss_weight,
            epoch_metrics=rec,
            history=history,
        )

    print(
        f"Done. best_val_mse={best_val:.6f} "
        f"best_val_count_mae={best_count:.6f}  saved to {run_dir}"
    )


if __name__ == "__main__":
    main()
