from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    Input:  x [B, F, T]
    Output: yhat [B, T]  (per-frame density regression)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        channels: list[int],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        pad = k // 2

        layers: list[nn.Module] = []
        c_in = int(in_channels)
        for c_out in channels:
            c_out = int(c_out)
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
            ]
            c_in = c_out

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(c_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        y = self.head(h).squeeze(1)  # [B, T]
        return F.softplus(y)
