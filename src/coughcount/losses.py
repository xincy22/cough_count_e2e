from __future__ import annotations

import torch


def _build_valid_mask(lengths: torch.Tensor, t: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(t, device=device)[None, :].expand(int(lengths.shape[0]), t)
    return idx < lengths[:, None]


def sample_masked_mse(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Per-sample frame MSE, shape [B]."""
    bsz, tsz = pred.shape
    mask = _build_valid_mask(lengths, tsz, pred.device).to(pred.dtype)
    err = (pred - target).pow(2) * mask
    denom = lengths.to(pred.dtype).clamp_min(1.0)
    return err.sum(dim=1) / denom


def sample_count_abs_error(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Per-sample count absolute error, shape [B]."""
    _, tsz = pred.shape
    mask = _build_valid_mask(lengths, tsz, pred.device)
    pred_count = (pred * mask).sum(dim=1)
    tgt_count = (target * mask).sum(dim=1)
    return (pred_count - tgt_count).abs()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(values.dtype).clamp_min(0.0)
    return (values * w).sum() / w.sum().clamp_min(1e-8)


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """
    pred/target: [B, T]
    lengths: [B]
    """
    each = sample_masked_mse(pred, target, lengths)
    return each.mean()


def masked_count_mae(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Differentiable count loss: MAE on per-window summed density."""
    return sample_count_abs_error(pred, target, lengths).mean()


def train_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    *,
    count_loss_weight: float = 0.1,
) -> torch.Tensor:
    """Training objective: frame MSE + λ * count MAE."""
    frame = masked_mse(pred, target, lengths)
    if float(count_loss_weight) <= 0.0:
        return frame
    count = masked_count_mae(pred, target, lengths)
    return frame + float(count_loss_weight) * count


def train_loss_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    *,
    count_loss_weight: float = 0.1,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weighted training objective.
    sample_weights: shape [B], typically for pos/neg rebalancing.
    """
    if sample_weights is None:
        return train_loss(
            pred,
            target,
            lengths,
            count_loss_weight=count_loss_weight,
        )

    frame_each = sample_masked_mse(pred, target, lengths)
    frame = weighted_mean(frame_each, sample_weights)
    if float(count_loss_weight) <= 0.0:
        return frame

    count_each = sample_count_abs_error(pred, target, lengths)
    count = weighted_mean(count_each, sample_weights)
    return frame + float(count_loss_weight) * count


@torch.no_grad()
def count_mae(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    return sample_count_abs_error(pred, target, lengths).mean()
