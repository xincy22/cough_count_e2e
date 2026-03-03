from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for k, v in override.items():
        if (
            k in merged
            and isinstance(merged[k], dict)
            and isinstance(v, dict)
        ):
            merged[k] = deep_merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_edgeai_config(
    *,
    base_config_path: Path,
    model_name: str | None = None,
    model_config_path: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = load_yaml_config(base_config_path)

    if model_config_path is not None:
        model_cfg = load_yaml_config(model_config_path)
        cfg = deep_merge_dict(cfg, model_cfg)
    elif model_name is not None and models_dir is not None:
        model_cfg_path = Path(models_dir) / f"{str(model_name).strip().lower()}.yaml"
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_cfg_path}")
        model_cfg = load_yaml_config(model_cfg_path)
        cfg = deep_merge_dict(cfg, model_cfg)

    if model_name is not None:
        cfg.setdefault("model", {})
        cfg["model"]["name"] = str(model_name).strip().lower()

    return cfg


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    try:
        tmp.replace(path)
    except PermissionError:
        # Windows 上目标文件被占用时，回退为直接写入，避免评估流程中断。
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        if tmp.exists():
            tmp.unlink()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tqdm_disabled() -> bool:
    """
    Whether tqdm progress bars should be disabled.

    Priority:
      1) COUGHCOUNT_TQDM in {"on","off"}
      2) auto-detect by TTY (disable when redirected to file/pipe)
    """
    mode = os.environ.get("COUGHCOUNT_TQDM", "auto").strip().lower()
    if mode in {"0", "false", "off", "disable", "disabled", "no"}:
        return True
    if mode in {"1", "true", "on", "enable", "enabled", "yes"}:
        return False
    try:
        return not bool(sys.stderr.isatty())
    except Exception:
        return True
