from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCounter(nn.Module):
    """
    GRU-only counting baseline.

    Input:  x [B, F, T]
    Output: yhat [B, T]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        proj_channels: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(proj_channels)

        # Pointwise projection keeps the baseline channel-compatible without
        # adding temporal convolution.
        self.in_proj = nn.Sequential(
            nn.Conv1d(int(in_channels), c, kernel_size=1, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

        self.gru = nn.GRU(
            input_size=c,
            hidden_size=int(gru_hidden),
            num_layers=int(gru_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=(float(dropout) if int(gru_layers) > 1 else 0.0),
        )

        head_in = int(gru_hidden) * (2 if bool(bidirectional) else 1)
        self.head = nn.Linear(head_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        y = self.head(h).squeeze(-1)
        return F.softplus(y)
