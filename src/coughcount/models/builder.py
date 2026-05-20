from __future__ import annotations

from typing import Any

from coughcount.models.bicrnn import BiCRNN
from coughcount.models.cnn1d import CNN1D
from coughcount.models.crnn import CRNN
from coughcount.models.dscnn import DSCNN
from coughcount.models.gru import GRUCounter
from coughcount.models.rescnn import ResCNN
from coughcount.models.tcn import TCN
from coughcount.models.tcn_attn import TCNAttn
from coughcount.models.tcn_gru import TCNGRU


def build_model(cfg: dict[str, Any], *, in_channels: int):
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "cnn1d")).lower()
    presets = model_cfg.get("presets", {})
    preset = dict(presets.get(name, {}))
    preset.pop("type", None)

    if name == "cnn1d":
        return CNN1D(in_channels=in_channels, **preset)
    if name == "tcn":
        return TCN(in_channels=in_channels, **preset)
    if name == "gru":
        return GRUCounter(in_channels=in_channels, **preset)
    if name == "crnn":
        return CRNN(in_channels=in_channels, **preset)
    if name == "bicrnn":
        return BiCRNN(in_channels=in_channels, **preset)
    if name == "tcn_gru":
        return TCNGRU(in_channels=in_channels, **preset)
    if name == "tcn_attn":
        return TCNAttn(in_channels=in_channels, **preset)
    if name == "dscnn":
        return DSCNN(in_channels=in_channels, **preset)
    if name == "rescnn":
        return ResCNN(in_channels=in_channels, **preset)

    raise KeyError(f"Unknown model.name: {name}")
