from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TCNBlock(nn.Module):
    def __init__(self, c: int, k: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (k // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(c, c, kernel_size=k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCN(nn.Module):
    """
    Input:  x [B, F, T]
    Output: yhat [B, T]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        channels: int = 256,
        layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()
        c = int(channels)
        self.in_proj = nn.Sequential(
            nn.Conv1d(int(in_channels), c, kernel_size=1, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )

        k = int(kernel_size)
        d_base = int(dilation_base)

        blocks: list[nn.Module] = []
        for i in range(int(layers)):
            d = d_base**i
            blocks.append(_TCNBlock(c, k, dilation=d, dropout=float(dropout)))

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        y = self.head(h).squeeze(1)
        return F.softplus(y)
