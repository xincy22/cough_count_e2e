from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from coughcount.data.dataset import EdgeAIWindowDataset, pad_collate
from coughcount.losses import masked_mse
from coughcount.models.builder import build_model


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def pick_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    pos_threshold: float,
) -> dict[str, float]:
    model.eval()

    mses: list[float] = []
    cmaes: list[float] = []

    cmae_pos: list[float] = []
    cmae_neg: list[float] = []
    mean_pred_neg: list[float] = []
    mean_gt_pos: list[float] = []
    
    pbar = tqdm(dl, desc="test", leave=False, dynamic_ncols=True)

    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)
        
        pred = model(x)

        mse = masked_mse(pred, y, lengths)
        mses.append(float(mse.item()))

        B, T = pred.shape
        idx = torch.arange(T, device=device)[None, :].expand(B, T)
        mask = idx < lengths[:, None]

        pred_count = (pred * mask).sum(dim=1)  # [B]
        gt_count = (y * mask).sum(dim=1)  # [B]
        mae_each = (pred_count - gt_count).abs()

        cmaes.extend(mae_each.cpu().tolist())

        is_pos = gt_count > float(pos_threshold)
        
        if bool(is_pos.any()):
            cmae_pos.extend(mae_each[is_pos].cpu().tolist())
            mean_gt_pos.extend(gt_count[is_pos].cpu().tolist())
        if bool((~is_pos).any()):
            cmae_neg.extend(mae_each[~is_pos].cpu().tolist())
            mean_pred_neg.extend(pred_count[~is_pos].cpu().tolist())
            
        pbar.set_postfix(
            mse=f"{np.mean(mses):.4f}",
            cmae=f"{np.mean(cmaes):.3f}",
        )

    out = {
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "count_mae": float(np.mean(cmaes)) if cmaes else float("nan"),
        "count_mae_pos": float(np.mean(cmae_pos)) if cmae_pos else float("nan"),
        "count_mae_neg": float(np.mean(cmae_neg)) if cmae_neg else float("nan"),
        "mean_pred_count_on_neg": (
            float(np.mean(mean_pred_neg)) if mean_pred_neg else float("nan")
        ),
        "mean_gt_count_on_pos": (
            float(np.mean(mean_gt_pos)) if mean_gt_pos else float("nan")
        ),
        "num_samples": len(cmaes),
        "num_pos": len(cmae_pos),
        "num_neg": len(cmae_neg),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained model on test set.")
    ap.add_argument("run_dir", type=Path, help="Path to the training run directory containing best.pt")
    
    args = ap.parse_args()

    split = "test"
    batch_size = 32
    device_name = "cuda"
    
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} does not exist")
    
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"Warning: {ckpt_path} not found. Checking last.pt...")
        ckpt_path = run_dir / "last.pt"
        if not ckpt_path.exists():
             raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    train_cfg = ckpt.get("cfg", {})
    if not train_cfg:
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            train_cfg = load_cfg(config_path)
    
    if not train_cfg:
        raise ValueError("Could not find configuration in checkpoint or run directory.")

    data_cfg = train_cfg.get("data", {})
    
    print(f"Loading {split} dataset...")
    ds_test = EdgeAIWindowDataset(
        split=split,
        mic=str(data_cfg.get("mic", "both")),
        window_sec=float(data_cfg.get("window_sec", 8.0)),
        hop_sec=float(data_cfg.get("hop_sec", 4.0)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.01)),
        return_meta=False,
    )
    
    print(f"Test dataset size: {len(ds_test)}")
    if len(ds_test) == 0:
        print("Warning: Dataset is empty.")
        return

    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_collate,
    )

    device = pick_device(device_name)
    print(f"Using device: {device}")

    in_channels = int(ds_test[0]["x"].shape[0])
    model = build_model(train_cfg, in_channels=in_channels)
    
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    
    print("Starting evaluation...")
    metrics = evaluate(model, dl_test, device, pos_threshold=float(data_cfg.get("pos_threshold", 0.01)))
    
    print("\nTest Results:")
    print(json.dumps(metrics, indent=2))
    
    out_file = run_dir / f"test_results_{split}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
