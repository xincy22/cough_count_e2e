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


class TCNGRU(nn.Module):
    """
    Input:  x [B, F, T]
    Output: yhat [B, T]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        tcn_channels: int = 192,
        tcn_layers: int = 6,
        tcn_kernel_size: int = 3,
        dilation_base: int = 2,
        gru_hidden: int = 192,
        gru_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(tcn_channels)

        self.in_proj = nn.Sequential(
            nn.Conv1d(int(in_channels), c, kernel_size=1, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for i in range(int(tcn_layers)):
            dilation = int(dilation_base) ** i
            blocks.append(
                _TCNBlock(
                    c,
                    k=int(tcn_kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                )
            )
        self.tcn = nn.Sequential(*blocks)

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
        h = self.tcn(h)
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        y = self.head(h).squeeze(-1)
        return F.softplus(y)
