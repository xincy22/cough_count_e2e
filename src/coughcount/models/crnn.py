from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CNN + GRU
    Input:  x [B, F, T]
    Output: yhat [B, T]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        cnn_channels: list[int],
        kernel_size: int = 5,
        rnn_hidden: int = 256,
        rnn_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        pad = k // 2

        layers: list[nn.Module] = []
        c_in = int(in_channels)
        for c_out in cnn_channels:
            c_out = int(c_out)
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
            ]
            c_in = c_out

        self.cnn = nn.Sequential(*layers)

        self.rnn = nn.GRU(
            input_size=c_in,
            hidden_size=int(rnn_hidden),
            num_layers=int(rnn_layers),
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(int(rnn_hidden), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)  # [B, C, T]
        h = h.transpose(1, 2)  # [B, T, C]
        h, _ = self.rnn(h)  # [B, T, H]
        return self.head(h).squeeze(-1)
