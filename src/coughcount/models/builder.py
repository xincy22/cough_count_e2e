from __future__ import annotations

from typing import Any

from coughcount.models.cnn1d import CNN1D
from coughcount.models.crnn import CRNN
from coughcount.models.tcn import TCN


def build_model(cfg: dict[str, Any], *, in_channels: int):
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "cnn1d")).lower()
    presets = model_cfg.get("presets", {})
    preset = dict(presets.get(name, {}))

    if name == "cnn1d":
        return CNN1D(in_channels=in_channels, **preset)
    if name == "tcn":
        return TCN(in_channels=in_channels, **preset)
    if name == "crnn":
        return CRNN(in_channels=in_channels, **preset)

    raise KeyError(f"Unknown model.name: {name}")
