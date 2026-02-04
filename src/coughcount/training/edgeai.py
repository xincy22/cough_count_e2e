from __future__ import annotations

import time
from dataclasses import dataclass
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
from coughcount.losses import masked_mse
from coughcount.models.builder import build_model
from coughcount.paths import ProjectPaths as P
from coughcount.utils.io import atomic_write_json


@dataclass(slots=True)
class TrainingComponents:
    train_dataset: EdgeAIWindowDataset
    val_dataset: EdgeAIWindowDataset
    train_loader: DataLoader
    val_loader: DataLoader
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: CosineAnnealingWarmRestarts
    epochs: int
    count_loss_weight: float
    pos_threshold: float
    cycle_epochs: int
    eta_min: float


@dataclass(slots=True)
class DynamicPosNegLossBalancer:
    enabled: bool
    pos_threshold: float
    ema_beta: float
    alpha: float
    min_pos_weight: float
    max_pos_weight: float
    warmup_epochs: int
    ema_pos_error: float | None = None
    ema_neg_error: float | None = None

    def current_pos_weight(self, epoch: int) -> float:
        if not self.enabled:
            return 1.0
        if int(epoch) <= int(self.warmup_epochs):
            return 1.0
        if self.ema_pos_error is None or self.ema_neg_error is None:
            return 1.0

        ratio = float(self.ema_pos_error) / max(float(self.ema_neg_error), 1e-8)
        weight = ratio ** float(self.alpha)
        weight = min(float(self.max_pos_weight), max(float(self.min_pos_weight), weight))
        return float(weight)

    def build_sample_weights(
        self,
        target: torch.Tensor,
        lengths: torch.Tensor,
        *,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        bsz, tsz = target.shape
        idx = torch.arange(tsz, device=target.device)[None, :].expand(bsz, tsz)
        mask = idx < lengths[:, None]
        gt_count = (target * mask).sum(dim=1)
        is_pos = gt_count > float(self.pos_threshold)

        pos_weight = self.current_pos_weight(epoch)
        weights = torch.ones_like(gt_count, dtype=torch.float32, device=target.device)
        if pos_weight != 1.0:
            weights = torch.where(is_pos, weights * float(pos_weight), weights)
        return weights, is_pos, pos_weight

    def update_from_batch_errors(
        self,
        sample_count_errors: torch.Tensor,
        is_pos: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        if bool(is_pos.any()):
            pos_err = float(sample_count_errors[is_pos].mean().item())
            if self.ema_pos_error is None:
                self.ema_pos_error = pos_err
            else:
                self.ema_pos_error = (
                    float(self.ema_beta) * float(self.ema_pos_error)
                    + (1.0 - float(self.ema_beta)) * pos_err
                )

        neg_mask = ~is_pos
        if bool(neg_mask.any()):
            neg_err = float(sample_count_errors[neg_mask].mean().item())
            if self.ema_neg_error is None:
                self.ema_neg_error = neg_err
            else:
                self.ema_neg_error = (
                    float(self.ema_beta) * float(self.ema_neg_error)
                    + (1.0 - float(self.ema_beta)) * neg_err
                )


def build_dynamic_pos_neg_loss_balancer(cfg: dict[str, Any]) -> DynamicPosNegLossBalancer:
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    dyn_cfg = train_cfg.get("dynamic_pos_neg_loss", {})
    if not isinstance(dyn_cfg, dict):
        dyn_cfg = {}

    return DynamicPosNegLossBalancer(
        enabled=bool(dyn_cfg.get("enabled", False)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        ema_beta=float(dyn_cfg.get("ema_beta", 0.95)),
        alpha=float(dyn_cfg.get("alpha", 0.5)),
        min_pos_weight=float(dyn_cfg.get("min_pos_weight", 1.0)),
        max_pos_weight=float(dyn_cfg.get("max_pos_weight", 6.0)),
        warmup_epochs=int(dyn_cfg.get("warmup_epochs", 1)),
    )


def prepare_training_components(
    cfg: dict[str, Any],
    *,
    device: torch.device,
) -> TrainingComponents:
    data_cfg = cfg.get("data", {})
    loader_cfg = cfg.get("loader", {})
    train_cfg = cfg.get("train", {})

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

    in_channels = int(ds_train[0]["x"].shape[0])
    model = build_model(cfg, in_channels=in_channels)
    assert isinstance(model, torch.nn.Module)
    model.to(device)

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("epochs", 10))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    cycle_epochs = max(1, int(train_cfg.get("lr_cycle_epochs", 10)))
    eta_min = float(train_cfg.get("lr_eta_min", 1e-6))
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cycle_epochs,
        T_mult=1,
        eta_min=eta_min,
    )

    return TrainingComponents(
        train_dataset=ds_train,
        val_dataset=ds_val,
        train_loader=dl_train,
        val_loader=dl_val,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        count_loss_weight=float(train_cfg.get("count_loss_weight", 0.1)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        cycle_epochs=cycle_epochs,
        eta_min=eta_min,
    )


def create_run_dir(cfg: dict[str, Any]) -> Path:
    run_name = time.strftime("%Y%m%d_%H%M%S")
    model_name = cfg.get("model", {}).get("name", "model")
    run_dir = P.runs / f"edgeai_{model_name}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    with (Path(run_dir) / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


@torch.no_grad()
def evaluate_counting_metrics(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    *,
    pos_threshold: float,
    desc: str = "val",
    per_sample: bool = False,
) -> dict[str, float]:
    model.eval()

    mses: list[float] = []
    cmaes: list[float] = []

    cmae_pos: list[float] = []
    cmae_neg: list[float] = []
    mean_pred_neg: list[float] = []
    mean_gt_pos: list[float] = []

    pbar = tqdm(dl, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)

        pred = model(x)

        mse = masked_mse(pred, y, lengths)
        mses.append(float(mse.item()))

        bsz, tsz = pred.shape
        idx = torch.arange(tsz, device=device)[None, :].expand(bsz, tsz)
        mask = idx < lengths[:, None]

        pred_count = (pred * mask).sum(dim=1)
        gt_count = (y * mask).sum(dim=1)
        mae_each = (pred_count - gt_count).abs()

        is_pos = gt_count > float(pos_threshold)

        if per_sample:
            cmaes.extend(mae_each.detach().cpu().tolist())
            if bool(is_pos.any()):
                cmae_pos.extend(mae_each[is_pos].detach().cpu().tolist())
                mean_gt_pos.extend(gt_count[is_pos].detach().cpu().tolist())
            if bool((~is_pos).any()):
                cmae_neg.extend(mae_each[~is_pos].detach().cpu().tolist())
                mean_pred_neg.extend(pred_count[~is_pos].detach().cpu().tolist())
        else:
            cmaes.append(float(mae_each.mean().item()))
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
    if per_sample:
        out["num_samples"] = int(len(cmaes))
        out["num_pos"] = int(len(cmae_pos))
        out["num_neg"] = int(len(cmae_neg))
    return out


def save_checkpoint(
    run_dir: Path,
    *,
    name: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "cfg": cfg,
        },
        Path(run_dir) / name,
    )


def _save_best_info(
    run_dir: Path,
    *,
    info_name: str,
    checkpoint_name: str,
    criterion: str,
    epoch_metrics: dict[str, Any],
    count_loss_weight: float,
) -> None:
    payload = dict(epoch_metrics)
    payload.update(
        {
            "checkpoint": checkpoint_name,
            "criterion": criterion,
            "count_loss_weight": float(count_loss_weight),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    atomic_write_json(Path(run_dir) / info_name, payload)


def save_epoch_artifacts(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_mse: float,
    val_count_mae: float,
    best_val_mse: float,
    best_val_count_mae: float,
    count_loss_weight: float,
    epoch_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[float, float]:
    best_val = float(best_val_mse)
    best_count = float(best_val_count_mae)

    if val_mse < best_val:
        best_val = float(val_mse)
        save_checkpoint(
            run_dir,
            name="best.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
        )
        _save_best_info(
            run_dir,
            info_name="best_info.json",
            checkpoint_name="best.pt",
            criterion="val_mse",
            epoch_metrics=epoch_metrics,
            count_loss_weight=count_loss_weight,
        )

    if val_count_mae < best_count:
        best_count = float(val_count_mae)
        save_checkpoint(
            run_dir,
            name="best_count.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
        )

    save_checkpoint(
        run_dir,
        name="last.pt",
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
    )
    atomic_write_json(Path(run_dir) / "history.json", history)

    return best_val, best_count
