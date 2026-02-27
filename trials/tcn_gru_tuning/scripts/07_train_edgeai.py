from __future__ import annotations

import os
import sys
from pathlib import Path

_TRIAL_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _TRIAL_ROOT.parent.parent
os.environ.setdefault("COUGHCOUNT_WORKSPACE", str(_TRIAL_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import argparse
import json
from pathlib import Path

import numpy as np
import torch
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


def _load_history(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else []


def _infer_best_values(
    history: list[dict[str, float | int]],
    *,
    selection_metric_key: str,
) -> tuple[float, float]:
    if not history:
        return float("inf"), float("inf")
    best_val = min(float(h.get("val_mse", float("inf"))) for h in history)
    best_count = min(
        float(
            h.get(
                selection_metric_key,
                h.get("val_count_mae", float("inf")),
            )
        )
        for h in history
    )
    return float(best_val), float(best_count)


def _resolve_selection_metric(
    train_cfg: dict,
    *,
    val_count_mae: float,
    val_count_mae_pos: float,
    val_count_mae_neg: float,
) -> tuple[str, float]:
    metric_name = str(train_cfg.get("best_metric", "val_count_mae")).lower()
    if metric_name == "val_weighted_count_mae":
        pos_w = float(train_cfg.get("val_metric_pos_weight", 1.0))
        neg_w = float(train_cfg.get("val_metric_neg_weight", 1.0))
        if pos_w < 0 or neg_w < 0 or (pos_w + neg_w) <= 0:
            raise ValueError(
                "train.val_metric_pos_weight and train.val_metric_neg_weight must be >=0 and not both zero."
            )
        if np.isnan(val_count_mae_pos) and np.isnan(val_count_mae_neg):
            return metric_name, float("inf")
        if np.isnan(val_count_mae_pos):
            return metric_name, float(val_count_mae_neg)
        if np.isnan(val_count_mae_neg):
            return metric_name, float(val_count_mae_pos)
        weighted = (
            pos_w * float(val_count_mae_pos) + neg_w * float(val_count_mae_neg)
        ) / (pos_w + neg_w)
        return metric_name, float(weighted)

    if metric_name == "val_count_mae_pos":
        return metric_name, float(val_count_mae_pos)
    if metric_name == "val_count_mae_neg":
        return metric_name, float(val_count_mae_neg)
    return "val_count_mae", float(val_count_mae)


def _enforce_equal_pos_neg_influence(
    sample_weights: torch.Tensor,
    is_pos: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """
    Reweight per-sample loss weights so that positive and negative groups
    contribute equally (50/50) to weighted losses within each batch.
    """
    if not enabled:
        return sample_weights

    pos_mask = is_pos.bool()
    neg_mask = ~pos_mask
    if not bool(pos_mask.any()) or not bool(neg_mask.any()):
        return sample_weights

    w = sample_weights.to(dtype=torch.float32)
    pos_sum = w[pos_mask].sum().clamp_min(1e-8)
    neg_sum = w[neg_mask].sum().clamp_min(1e-8)
    w = w.clone()
    w[pos_mask] = w[pos_mask] * (0.5 / pos_sum)
    w[neg_mask] = w[neg_mask] * (0.5 / neg_sum)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=P.configs / "edgeai.yaml")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-ckpt", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    if args.resume and args.run_dir is not None and (args.run_dir / "config.yaml").exists():
        cfg = load_yaml_config(args.run_dir / "config.yaml")
    set_seed(int(cfg.get("seed", 0)))

    train_cfg = cfg.get("train", {})
    device = pick_device(train_cfg.get("device", "cuda"))
    under_count_weight = float(train_cfg.get("under_count_weight", 0.0))
    under_count_positive_only = bool(train_cfg.get("under_count_positive_only", True))
    equal_pos_neg_influence = bool(train_cfg.get("equal_pos_neg_influence", False))
    print(f"Device: {device}")
    best_metric_key = str(train_cfg.get("best_metric", "val_count_mae")).lower()

    components = prepare_training_components(cfg, device=device)
    loss_balancer = build_dynamic_pos_neg_loss_balancer(cfg)

    if args.init_ckpt is not None and not args.resume:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        components.model.load_state_dict(ckpt["model_state"])
        print(f"Initialized model weights from: {args.init_ckpt}")

    run_dir = args.run_dir if args.run_dir is not None else create_run_dir(cfg)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir, cfg)

    print(f"Run dir: {run_dir}")
    print(
        f"Train windows: {len(components.train_dataset)} "
        f"(pos={len(components.train_dataset.pos_idx)} neg={len(components.train_dataset.neg_idx)})"
    )
    print(
        f"Val windows:   {len(components.val_dataset)} "
        f"(pos={components.val_pos_windows} neg={components.val_neg_windows})"
        f"{' [balanced 1:1]' if components.val_is_balanced else ''}"
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
            f"min_pos_ratio={loss_balancer.min_pos_ratio:g}, "
            f"max_pos_ratio={loss_balancer.max_pos_ratio:g}, "
            f"warmup_epochs={loss_balancer.warmup_epochs})"
        )
    if under_count_weight > 0.0:
        print(
            "Under-count loss: "
            f"enabled (weight={under_count_weight:g}, "
            f"positive_only={under_count_positive_only}, "
            f"pos_threshold={components.pos_threshold:g})"
        )
    if equal_pos_neg_influence and not loss_balancer.enabled:
        print("Equal pos/neg loss influence: enabled (50/50 per batch)")
    elif equal_pos_neg_influence and loss_balancer.enabled:
        print(
            "Equal pos/neg loss influence: ignored because dynamic_pos_neg_loss is enabled "
            "(using dynamic a*pos + (1-a)*neg)."
        )

    history_path = run_dir / "history.json"
    history = _load_history(history_path)
    history_metric_key = (
        "val_weighted_count_mae"
        if best_metric_key == "val_weighted_count_mae"
        else best_metric_key
    )
    best_val, best_count = _infer_best_values(
        history,
        selection_metric_key=history_metric_key,
    )
    start_epoch = 1

    if args.resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu")
        components.model.load_state_dict(ckpt["model_state"])
        components.optimizer.load_state_dict(ckpt["opt_state"])
        if ckpt.get("scheduler_state") is not None:
            components.scheduler.load_state_dict(ckpt["scheduler_state"])
        lb_state = ckpt.get("loss_balancer_state")
        if isinstance(lb_state, dict):
            loss_balancer.ema_pos_error = lb_state.get("ema_pos_error")
            loss_balancer.ema_neg_error = lb_state.get("ema_neg_error")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resume from epoch {start_epoch}")

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
            sample_weights = _enforce_equal_pos_neg_influence(
                sample_weights,
                is_pos,
                enabled=equal_pos_neg_influence and (not loss_balancer.enabled),
            )

            pred = components.model(x)
            loss = train_loss_weighted(
                pred,
                y,
                lengths,
                count_loss_weight=components.count_loss_weight,
                sample_weights=sample_weights if loss_balancer.enabled else None,
                under_count_weight=under_count_weight,
                under_count_positive_only=under_count_positive_only,
                pos_threshold=components.pos_threshold,
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
        metric_name, metric_value = _resolve_selection_metric(
            train_cfg,
            val_count_mae=rec["val_count_mae"],
            val_count_mae_pos=rec["val_count_mae_pos"],
            val_count_mae_neg=rec["val_count_mae_neg"],
        )
        rec["val_selection_metric_name"] = metric_name
        rec["val_selection_metric"] = float(metric_value)
        if metric_name == "val_weighted_count_mae":
            rec["val_metric_pos_weight"] = float(train_cfg.get("val_metric_pos_weight", 1.0))
            rec["val_metric_neg_weight"] = float(train_cfg.get("val_metric_neg_weight", 1.0))
            rec["val_weighted_count_mae"] = float(metric_value)
        history.append(rec)

        print(
            f"[epoch {epoch}] "
            f"lr={lr_now:.2e} "
            f"train_mse={train_mse:.6f} train_count_mae={train_cmae:.4f} "
            f"val_mse={val_mse:.6f} val_count_mae={val_cmae:.4f} "
            f"sel={metric_name}:{metric_value:.4f} "
            f"(pos_mae={val_stats['count_mae_pos']:.3f} neg_mae={val_stats['count_mae_neg']:.3f} "
            f"neg_pred={val_stats['mean_pred_count_on_neg']:.2f})"
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
            extra_state={
                "loss_balancer_state": {
                    "ema_pos_error": loss_balancer.ema_pos_error,
                    "ema_neg_error": loss_balancer.ema_neg_error,
                }
            },
            selection_metric_name=metric_name,
            selection_metric_value=float(metric_value),
        )

    print(
        f"Done. best_val_mse={best_val:.6f} "
        f"best_val_count_mae={best_count:.6f}  saved to {run_dir}"
    )


if __name__ == "__main__":
    main()
