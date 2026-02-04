from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from coughcount.data.dataset import EdgeAIWindowDataset, pad_collate
from coughcount.models.builder import build_model
from coughcount.training.edgeai import evaluate_counting_metrics
from coughcount.utils.config import load_yaml_config
from coughcount.utils.io import atomic_write_json
from coughcount.utils.runtime import pick_device


def resolve_checkpoint_path(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    best_path = run_dir / "best.pt"
    if best_path.exists():
        return best_path

    last_path = run_dir / "last.pt"
    if last_path.exists():
        return last_path

    raise FileNotFoundError(f"No checkpoint found in {run_dir}. Expected best.pt or last.pt")


def load_run_config(run_dir: Path, ckpt: dict[str, Any]) -> dict[str, Any]:
    cfg = ckpt.get("cfg", {})
    if isinstance(cfg, dict) and cfg:
        return cfg

    config_path = Path(run_dir) / "config.yaml"
    if config_path.exists():
        return load_yaml_config(config_path)

    raise ValueError("Could not find configuration in checkpoint or run directory.")


def build_eval_dataloader(
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    num_workers: int,
) -> tuple[EdgeAIWindowDataset, DataLoader]:
    data_cfg = cfg.get("data", {})

    ds = EdgeAIWindowDataset(
        split=split,
        mic=str(data_cfg.get("mic", "both")),
        window_sec=float(data_cfg.get("window_sec", 8.0)),
        hop_sec=float(data_cfg.get("hop_sec", 4.0)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        return_meta=False,
    )

    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=pad_collate,
        pin_memory=False,
        persistent_workers=(int(num_workers) > 0),
    )
    return ds, dl


def evaluate_run_on_split(
    run_dir: Path,
    *,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    device_name: str = "cuda",
) -> tuple[dict[str, float], Path, Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} does not exist")

    ckpt_path = resolve_checkpoint_path(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = load_run_config(run_dir, ckpt)

    ds, dl = build_eval_dataloader(
        cfg,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if len(ds) == 0:
        raise RuntimeError(f"{split} dataset is empty.")

    device = pick_device(device_name)
    in_channels = int(ds[0]["x"].shape[0])
    model = build_model(cfg, in_channels=in_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    metrics = evaluate_counting_metrics(
        model,
        dl,
        device,
        pos_threshold=float(cfg.get("data", {}).get("pos_threshold", 0.01)),
        desc=split,
        per_sample=True,
    )

    out_file = run_dir / f"test_results_{split}.json"
    atomic_write_json(out_file, metrics)
    return metrics, out_file, ckpt_path
