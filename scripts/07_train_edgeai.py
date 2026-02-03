from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from coughcount.data.dataset import EdgeAIWindowDataset, pad_collate
from coughcount.data.sampling import BalancedSampler
from coughcount.losses import count_mae, masked_mse, train_loss
from coughcount.models.builder import build_model
from coughcount.paths import ProjectPaths as P


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def pick_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    *,
    pos_threshold: float,
) -> dict[str, float]:
    model.eval()

    mses: list[float] = []
    cmaes: list[float] = []

    cmae_pos: list[float] = []
    cmae_neg: list[float] = []
    mean_pred_neg: list[float] = []
    mean_gt_pos: list[float] = []

    pbar = tqdm(dl, desc="val", leave=False, dynamic_ncols=True)

    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)

        pred = model(x)

        mse = masked_mse(pred, y, lengths)
        mses.append(float(mse.item()))

        # build mask
        B, T = pred.shape
        idx = torch.arange(T, device=device)[None, :].expand(B, T)
        mask = idx < lengths[:, None]

        pred_count = (pred * mask).sum(dim=1)  # [B]
        gt_count = (y * mask).sum(dim=1)  # [B]
        mae_each = (pred_count - gt_count).abs()

        cmaes.append(float(mae_each.mean().item()))

        is_pos = gt_count > float(pos_threshold)

        if bool(is_pos.any()):
            cmae_pos.append(float(mae_each[is_pos].mean().item()))
            mean_gt_pos.append(float(gt_count[is_pos].mean().item()))
        if bool((~is_pos).any()):
            cmae_neg.append(float(mae_each[~is_pos].mean().item()))
            mean_pred_neg.append(float(pred_count[~is_pos].mean().item()))

        pbar.set_postfix(
            mse=f"{np.mean(mses):.4f}",
            cmae=f"{np.mean(cmaes):.3f}",
            neg_pred=f"{np.mean(mean_pred_neg):.2f}" if mean_pred_neg else "nan",
        )

    out = {
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "count_mae": float(np.mean(cmaes)) if cmaes else float("nan"),
        "count_mae_pos": float(np.mean(cmae_pos)) if cmae_pos else float("nan"),
        "count_mae_neg": float(np.mean(cmae_neg)) if cmae_neg else float("nan"),
        "mean_pred_count_on_neg": (
            float(np.mean(mean_pred_neg)) if mean_pred_neg else float("nan")
        ),
        "mean_gt_count_on_pos": (
            float(np.mean(mean_gt_pos)) if mean_gt_pos else float("nan")
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=P.configs / "edgeai.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 0)))

    data_cfg = cfg.get("data", {})
    loader_cfg = cfg.get("loader", {})
    train_cfg = cfg.get("train", {})

    device = pick_device(train_cfg.get("device", "cuda"))
    print(f"Device: {device}")

    ds_train = EdgeAIWindowDataset(
        split=str(data_cfg.get("split_train", "train")),
        mic=str(data_cfg.get("mic", "both")),
        window_sec=float(data_cfg.get("window_sec", 8.0)),
        hop_sec=float(data_cfg.get("hop_sec", 4.0)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        return_meta=False,
    )
    ds_val = EdgeAIWindowDataset(
        split=str(data_cfg.get("split_val", "val")),
        mic=str(data_cfg.get("mic", "both")),
        window_sec=float(data_cfg.get("window_sec", 8.0)),
        hop_sec=float(data_cfg.get("hop_sec", 4.0)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        return_meta=False,
    )

    batch_size = int(loader_cfg.get("batch_size", 16))
    num_workers = int(loader_cfg.get("num_workers", 4))
    pos_frac = float(loader_cfg.get("pos_fraction", 0.5))

    train_sampler = BalancedSampler(
        ds_train.pos_idx,
        ds_train.neg_idx,
        batch_size=batch_size,
        pos_fraction=pos_frac,
        seed=int(cfg.get("seed", 0)),
    )

    dl_train = DataLoader(
        ds_train,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    in_channels = int(ds_train[0]["x"].shape[0])  # F=513
    model = build_model(cfg, in_channels=in_channels)
    assert isinstance(model, torch.nn.Module)
    model.to(device)

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("epochs", 10))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    cycle_epochs = int(train_cfg.get("lr_cycle_epochs", 10))
    cycle_epochs = max(1, cycle_epochs)
    eta_min = float(train_cfg.get("lr_eta_min", 1e-6))

    scheduler = CosineAnnealingWarmRestarts(
        opt,
        T_0=cycle_epochs,
        T_mult=1,
        eta_min=eta_min,
    )

    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = P.runs / f"edgeai_{cfg.get('model', {}).get('name', 'model')}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"Run dir: {run_dir}")
    print(
        f"Train windows: {len(ds_train)} (pos={len(ds_train.pos_idx)} neg={len(ds_train.neg_idx)})"
    )
    print(
        f"Val windows:   {len(ds_val)} (pos={len(ds_val.pos_idx)} neg={len(ds_val.neg_idx)})"
    )
    print(
        f"Model: {cfg['model']['name']} preset={cfg['model']['presets'][cfg['model']['name']]}"
    )
    print(
        f"LR schedule: CosineAnnealingWarmRestarts(T_0={cycle_epochs}, eta_min={eta_min:g})"
    )

    best_val = float("inf")
    history: list[dict[str, Any]] = []

    pos_threshold = float(data_cfg.get("pos_threshold", 0.01))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []

        count_loss_weight = float(train_cfg.get("count_loss_weight", 0.1))

        pbar = tqdm(dl_train, desc=f"train e{epoch}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)

            pred = model(x)
            loss = train_loss(pred, y, lengths, count_loss_weight=count_loss_weight)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.item()))
            with torch.no_grad():
                train_maes.append(float(count_mae(pred, y, lengths).item()))

            epoch_progress = (epoch - 1) + (pbar.n / max(1, len(dl_train)))
            scheduler.step(epoch_progress)

            lr_now = float(opt.param_groups[0]["lr"])
            pbar.set_postfix(
                mse=f"{np.mean(train_losses):.4f}",
                cmae=f"{np.mean(train_maes):.3f}",
                lr=f"{lr_now:.2e}",
            )

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        train_cmae = float(np.mean(train_maes)) if train_maes else float("nan")

        val_stats = evaluate(
            model,
            dl_val,
            device,
            pos_threshold=pos_threshold,
        )
        val_mse = float(val_stats["mse"])
        val_cmae = float(val_stats["count_mae"])
        lr_now = float(opt.param_groups[0]["lr"])

        rec = {
            "epoch": epoch,
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

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "cfg": cfg,
                },
                run_dir / "best.pt",
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "cfg": cfg,
            },
            run_dir / "last.pt",
        )
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Done. best_val_mse={best_val:.6f}  saved to {run_dir}")


if __name__ == "__main__":
    main()
