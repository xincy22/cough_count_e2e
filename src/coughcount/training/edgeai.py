from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Subset
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
    val_dataset: Dataset
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
    val_pos_windows: int
    val_neg_windows: int
    val_is_balanced: bool


@dataclass(slots=True)
class DynamicPosNegLossBalancer:
    enabled: bool
    pos_threshold: float
    ema_beta: float
    alpha: float
    min_pos_ratio: float
    max_pos_ratio: float
    warmup_epochs: int
    ema_pos_error: float | None = None
    ema_neg_error: float | None = None

    def current_pos_ratio(self, epoch: int) -> float:
        if not self.enabled:
            return 0.5
        if int(epoch) <= int(self.warmup_epochs):
            return float(
                min(
                    max(0.5, float(self.min_pos_ratio)),
                    float(self.max_pos_ratio),
                )
            )
        if self.ema_pos_error is None or self.ema_neg_error is None:
            return float(
                min(
                    max(0.5, float(self.min_pos_ratio)),
                    float(self.max_pos_ratio),
                )
            )

        denom = max(float(self.ema_pos_error) + float(self.ema_neg_error), 1e-8)
        raw = float(self.ema_pos_error) / denom  # [0, 1]
        raw = min(max(raw, 1e-6), 1.0 - 1e-6)

        # alpha controls how aggressively we move away from 0.5
        sharp = max(float(self.alpha), 1e-6)
        p = raw**sharp
        q = (1.0 - raw) ** sharp
        ratio = p / max(p + q, 1e-8)

        ratio = min(float(self.max_pos_ratio), max(float(self.min_pos_ratio), ratio))
        return float(ratio)

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

        pos_ratio = self.current_pos_ratio(epoch)
        weights = torch.ones_like(gt_count, dtype=torch.float32, device=target.device)
        if not self.enabled:
            return weights, is_pos, pos_ratio

        pos_mask = is_pos.bool()
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

        if n_pos > 0 and n_neg > 0:
            weights = torch.zeros_like(weights)
            weights[pos_mask] = float(pos_ratio) / float(n_pos)
            weights[neg_mask] = float(1.0 - pos_ratio) / float(n_neg)
        return weights, is_pos, pos_ratio

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

    min_pos_ratio = float(dyn_cfg.get("min_pos_ratio", 0.5))
    max_pos_ratio = float(dyn_cfg.get("max_pos_ratio", 0.5))
    if not (0.0 <= min_pos_ratio <= 1.0 and 0.0 <= max_pos_ratio <= 1.0):
        raise ValueError("dynamic_pos_neg_loss min_pos_ratio/max_pos_ratio must be within [0, 1].")
    if min_pos_ratio > max_pos_ratio:
        raise ValueError("dynamic_pos_neg_loss min_pos_ratio must be <= max_pos_ratio.")

    return DynamicPosNegLossBalancer(
        enabled=bool(dyn_cfg.get("enabled", False)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        ema_beta=float(dyn_cfg.get("ema_beta", 0.95)),
        alpha=float(dyn_cfg.get("alpha", 0.5)),
        min_pos_ratio=min_pos_ratio,
        max_pos_ratio=max_pos_ratio,
        warmup_epochs=int(dyn_cfg.get("warmup_epochs", 1)),
    )


def _build_equal_pos_neg_val_dataset(
    ds_val: EdgeAIWindowDataset,
    *,
    seed: int,
) -> tuple[Dataset, int, int, bool]:
    pos = np.asarray(ds_val.pos_idx, dtype=np.int64)
    neg = np.asarray(ds_val.neg_idx, dtype=np.int64)
    if pos.size == 0 or neg.size == 0:
        return ds_val, int(pos.size), int(neg.size), False

    n = int(min(pos.size, neg.size))
    rng = np.random.default_rng(int(seed))
    if pos.size > n:
        pos = rng.choice(pos, size=n, replace=False)
    if neg.size > n:
        neg = rng.choice(neg, size=n, replace=False)

    indices = np.concatenate([pos, neg]).astype(np.int64)
    indices.sort()
    return Subset(ds_val, indices.tolist()), n, n, True


def prepare_training_components(
    cfg: dict[str, Any],
    *,
    device: torch.device,
) -> TrainingComponents:
    data_cfg = cfg.get("data", {})
    loader_cfg = cfg.get("loader", {})
    train_cfg = cfg.get("train", cfg.get("training", {}))

    # Resolve splits_json path - handle both absolute and relative paths
    # Relative paths in experiment configs are typically relative to the experiment dir
    splits_json_input = data_cfg.get("splits_json", P.edgeai_splits_json)
    splits_json = Path(str(splits_json_input))
    if not splits_json.is_absolute():
        # Resolve from current working directory (typically experiment dir)
        # If that doesn't exist, fall back to project root
        resolved_from_cwd = Path(splits_json).resolve()
        if resolved_from_cwd.exists():
            splits_json = resolved_from_cwd
        else:
            # Try resolving from project root (for configs without parent refs)
            splits_json = (P.root / splits_json).resolve()

    # Get npy_dir from config if specified, otherwise use default
    npy_dir_input = data_cfg.get("npy_dir", P.edgeai_npy)
    npy_dir = Path(str(npy_dir_input))
    if not npy_dir.is_absolute():
        # Resolve from current working directory (typically experiment dir)
        resolved_from_cwd = Path(npy_dir).resolve()
        if resolved_from_cwd.exists():
            npy_dir = resolved_from_cwd
        else:
            # Try resolving from project root (for configs without parent refs)
            npy_dir = (P.root / npy_dir).resolve()

    ds_train = EdgeAIWindowDataset(
        split=str(data_cfg.get("split_train", "train")),
        npy_dir=npy_dir,
        splits_json=splits_json,
        mic=str(data_cfg.get("mic", "both")),
        window_sec=float(data_cfg.get("window_sec", 8.0)),
        hop_sec=float(data_cfg.get("hop_sec", 4.0)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        return_meta=False,
    )
    ds_val = EdgeAIWindowDataset(
        split=str(data_cfg.get("split_val", "val")),
        npy_dir=npy_dir,
        splits_json=splits_json,
        mic=str(data_cfg.get("mic", "both")),
        window_sec=float(data_cfg.get("window_sec", 8.0)),
        hop_sec=float(data_cfg.get("hop_sec", 4.0)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        return_meta=False,
    )

    batch_size = int(loader_cfg.get("batch_size", 16))
    num_workers = int(loader_cfg.get("num_workers", 4))
    pos_frac = float(loader_cfg.get("pos_fraction", 0.5))
    val_equal_pos_neg = bool(loader_cfg.get("val_equal_pos_neg", False))
    val_balance_seed = int(loader_cfg.get("val_balance_seed", int(cfg.get("seed", 0))))

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

    ds_val_eval: Dataset = ds_val
    val_pos_windows = int(len(ds_val.pos_idx))
    val_neg_windows = int(len(ds_val.neg_idx))
    val_is_balanced = False
    if val_equal_pos_neg:
        ds_val_eval, val_pos_windows, val_neg_windows, val_is_balanced = (
            _build_equal_pos_neg_val_dataset(
                ds_val,
                seed=val_balance_seed,
            )
        )

    dl_val = DataLoader(
        ds_val_eval,
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
        val_dataset=ds_val_eval,
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
        val_pos_windows=val_pos_windows,
        val_neg_windows=val_neg_windows,
        val_is_balanced=val_is_balanced,
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

    # MSE is aggregated as mean over batches (kept for backward comparability).
    mses: list[float] = []

    # Count MAE and breakdowns are aggregated globally per-sample so that:
    # - val_count_mae is a true dataset-level mean absolute count error
    # - if val is balanced 1:1 (pos/neg by sample), val_count_mae matches (pos_mae+neg_mae)/2
    cmae_sum = 0.0
    cmae_n = 0

    cmae_pos_sum = 0.0
    cmae_pos_n = 0
    cmae_neg_sum = 0.0
    cmae_neg_n = 0

    pred_neg_sum = 0.0
    gt_pos_sum = 0.0

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

        # Always aggregate per-sample globally; keep `per_sample` in signature to avoid
        # forcing changes in callers/configs.
        cmae_sum += float(mae_each.sum().item())
        cmae_n += int(mae_each.numel())
        if bool(is_pos.any()):
            pos_err = mae_each[is_pos]
            cmae_pos_sum += float(pos_err.sum().item())
            cmae_pos_n += int(pos_err.numel())
            gt_pos_sum += float(gt_count[is_pos].sum().item())
        if bool((~is_pos).any()):
            neg_err = mae_each[~is_pos]
            cmae_neg_sum += float(neg_err.sum().item())
            cmae_neg_n += int(neg_err.numel())
            pred_neg_sum += float(pred_count[~is_pos].sum().item())

        pbar.set_postfix(
            mse=f"{np.mean(mses):.4f}",
            cmae=f"{(cmae_sum / max(1, cmae_n)):.3f}",
            neg_pred=f"{(pred_neg_sum / max(1, cmae_neg_n)):.2f}"
            if cmae_neg_n
            else "nan",
        )

    out = {
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "count_mae": float(cmae_sum / max(1, cmae_n)) if cmae_n else float("nan"),
        "count_mae_pos": (
            float(cmae_pos_sum / max(1, cmae_pos_n)) if cmae_pos_n else float("nan")
        ),
        "count_mae_neg": (
            float(cmae_neg_sum / max(1, cmae_neg_n)) if cmae_neg_n else float("nan")
        ),
        "mean_pred_count_on_neg": (
            float(pred_neg_sum / max(1, cmae_neg_n)) if cmae_neg_n else float("nan")
        ),
        "mean_gt_count_on_pos": (
            float(gt_pos_sum / max(1, cmae_pos_n)) if cmae_pos_n else float("nan")
        ),
    }
    # Provide counts unconditionally (useful for debugging split/balancing).
    out["num_samples"] = int(cmae_n)
    out["num_pos"] = int(cmae_pos_n)
    out["num_neg"] = int(cmae_neg_n)
    return out


def save_checkpoint(
    run_dir: Path,
    *,
    name: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmRestarts | None = None,
    cfg: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "cfg": cfg,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra_state:
        payload.update(extra_state)

    torch.save(
        payload,
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
    scheduler: CosineAnnealingWarmRestarts | None,
    val_mse: float,
    val_count_mae: float,
    best_val_mse: float,
    best_val_count_mae: float,
    count_loss_weight: float,
    epoch_metrics: dict[str, Any],
    history: list[dict[str, Any]],
    extra_state: dict[str, Any] | None = None,
    selection_metric_name: str = "val_count_mae",
    selection_metric_value: float | None = None,
) -> tuple[float, float]:
    best_val = float(best_val_mse)
    best_count = float(best_val_count_mae)
    metric_name = str(selection_metric_name)
    metric_value = (
        float(selection_metric_value)
        if selection_metric_value is not None
        else float(val_count_mae)
    )

    if metric_value < best_count:
        best_count = float(metric_value)
        save_checkpoint(
            run_dir,
            name="best.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            extra_state=extra_state,
        )
        _save_best_info(
            run_dir,
            info_name="best_info.json",
            checkpoint_name="best.pt",
            criterion=metric_name,
            epoch_metrics=epoch_metrics,
            count_loss_weight=count_loss_weight,
        )
        save_checkpoint(
            run_dir,
            name="best_count.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            extra_state=extra_state,
        )

    if val_mse < best_val:
        best_val = float(val_mse)
        save_checkpoint(
            run_dir,
            name="best_mse.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            extra_state=extra_state,
        )

    save_checkpoint(
        run_dir,
        name="last.pt",
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        extra_state=extra_state,
    )
    atomic_write_json(Path(run_dir) / "history.json", history)

    return best_val, best_count
