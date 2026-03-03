from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        k = int(kernel_size)
        pad = k // 2

        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(float(dropout))

        if c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm1d(c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.drop(h)

        h = h + self.shortcut(x)
        return F.relu(h, inplace=True)


class ResCNN(nn.Module):
    """
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
                _ResBlock1D(
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
