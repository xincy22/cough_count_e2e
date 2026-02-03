from __future__ import annotations

import torch
import torch.nn as nn


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """
    pred/target: [B, T]
    lengths: [B]
    """
    B, T = pred.shape
    idx = torch.arange(T, device=pred.device)[None, :].expand(B, T)
    mask = idx < lengths[:, None]
    return nn.functional.mse_loss(pred[mask], target[mask])


def masked_count_mae(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Differentiable count loss: MAE on per-window summed density."""
    B, T = pred.shape
    idx = torch.arange(T, device=pred.device)[None, :].expand(B, T)
    mask = idx < lengths[:, None]
    pred_count = (pred * mask).sum(dim=1)
    tgt_count = (target * mask).sum(dim=1)
    return (pred_count - tgt_count).abs().mean()


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


@torch.no_grad()
def count_mae(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    B, T = pred.shape
    idx = torch.arange(T, device=pred.device)[None, :].expand(B, T)
    mask = idx < lengths[:, None]
    pred_count = (pred * mask).sum(dim=1)
    tgt_count = (target * mask).sum(dim=1)
    return (pred_count - tgt_count).abs().mean()
