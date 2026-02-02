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
