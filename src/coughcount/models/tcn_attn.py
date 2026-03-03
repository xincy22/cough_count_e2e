from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TCNResidualBlock(nn.Module):
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
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class _SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float, ff_mult: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(channels * ff_mult, channels),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class TCNAttn(nn.Module):
    """
    Input:  x [B, F, T]
    Output: yhat [B, T]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        channels: int = 192,
        tcn_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
        attn_heads: int = 4,
        attn_layers: int = 1,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        c = int(channels)
        heads = int(attn_heads)
        if c % heads != 0:
            raise ValueError(
                f"channels must be divisible by attn_heads, got channels={c}, heads={heads}"
            )

        self.in_proj = nn.Sequential(
            nn.Conv1d(int(in_channels), c, kernel_size=1, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for i in range(int(tcn_layers)):
            blocks.append(
                _TCNResidualBlock(
                    c=c,
                    k=int(kernel_size),
                    dilation=int(dilation_base) ** i,
                    dropout=float(dropout),
                )
            )
        self.tcn = nn.Sequential(*blocks)

        self.attn = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    channels=c,
                    heads=heads,
                    dropout=float(dropout),
                    ff_mult=int(ff_mult),
                )
                for _ in range(int(attn_layers))
            ]
        )
        self.head = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.tcn(h)
        h = h.transpose(1, 2)  # [B, T, C]
        for blk in self.attn:
            h = blk(h)
        y = self.head(h).squeeze(-1)
        return F.softplus(y)
