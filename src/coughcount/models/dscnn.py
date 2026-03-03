from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DSConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=k, padding=pad, groups=c_in, bias=False),
            nn.BatchNorm1d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad, groups=c_out, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.Dropout(float(dropout)),
        )
        if c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm1d(c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.shortcut(x), inplace=True)


class DSCNN(nn.Module):
    """
    Depthwise-separable residual CNN.
    Input:  x [B, F, T]
    Output: yhat [B, T]
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
        if not channels:
            raise ValueError("channels must not be empty.")

        blocks: list[nn.Module] = []
        c_in = int(in_channels)
        for c_out in channels:
            c_out = int(c_out)
            blocks.append(
                _DSConvBlock(
                    c_in=c_in,
                    c_out=c_out,
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
            )
            c_in = c_out

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(c_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        y = self.head(h).squeeze(1)
        return F.softplus(y)

