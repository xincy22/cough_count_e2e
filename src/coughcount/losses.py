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


def sample_count_under_error(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Per-sample count underestimation error max(gt_count - pred_count, 0), shape [B]."""
    _, tsz = pred.shape
    mask = _build_valid_mask(lengths, tsz, pred.device)
    pred_count = (pred * mask).sum(dim=1)
    tgt_count = (target * mask).sum(dim=1)
    return (tgt_count - pred_count).relu()


def sample_target_count(
    target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Per-sample target count, shape [B]."""
    _, tsz = target.shape
    mask = _build_valid_mask(lengths, tsz, target.device)
    return (target * mask).sum(dim=1)


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
    under_count_weight: float = 0.0,
    under_count_positive_only: bool = True,
    pos_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Weighted training objective.
    sample_weights: shape [B], typically for pos/neg rebalancing.
    """
    if sample_weights is None:
        sample_weights = torch.ones(
            pred.shape[0], device=pred.device, dtype=pred.dtype
        )

    frame_each = sample_masked_mse(pred, target, lengths)
    frame = weighted_mean(frame_each, sample_weights)
    loss = frame

    if float(count_loss_weight) > 0.0:
        count_each = sample_count_abs_error(pred, target, lengths)
        count = weighted_mean(count_each, sample_weights)
        loss = loss + float(count_loss_weight) * count

    if float(under_count_weight) > 0.0:
        under_each = sample_count_under_error(pred, target, lengths)
        if bool(under_count_positive_only):
            tgt_count = sample_target_count(target, lengths)
            pos_mask = (tgt_count > float(pos_threshold)).to(under_each.dtype)
            under_weights = sample_weights * pos_mask
        else:
            under_weights = sample_weights
        under = weighted_mean(under_each, under_weights)
        loss = loss + float(under_count_weight) * under

    return loss


@torch.no_grad()
def count_mae(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    return sample_count_abs_error(pred, target, lengths).mean()
