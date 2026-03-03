from __future__ import annotations

from pathlib import Path

import yaml

from torch.utils.data import DataLoader

from coughcount.data.dataset import EdgeAIWindowDataset, pad_collate
from coughcount.data.sampling import BalancedSampler


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> None:
    from pathlib import Path

    cfg = load_config()
    dl_cfg = cfg["dataloader_check"]

    split = dl_cfg["split"]
    mic = dl_cfg["mic"]
    window_sec = dl_cfg["window_sec"]
    hop_sec = dl_cfg["hop_sec"]
    pos_threshold = dl_cfg["pos_threshold"]
    batch_size = dl_cfg["batch_size"]
    pos_frac = dl_cfg["pos_frac"]
    num_workers = dl_cfg["num_workers"]
    max_batches = dl_cfg["max_batches"]

    ds = EdgeAIWindowDataset(
        split=split,
        mic=mic,
        window_sec=window_sec,
        hop_sec=hop_sec,
        pos_threshold=pos_threshold,
        return_meta=True,
    )

    print("Dataset OK")
    print(f"  samples: {len(ds.samples)}")
    print(f"  windows: {len(ds)}")
    print(f"  pos windows: {len(ds.pos_idx)}")
    print(f"  neg windows: {len(ds.neg_idx)}")
    if len(ds.pos_idx) > 0:
        print(f"  pos ratio: {len(ds.pos_idx) / max(1, len(ds)):.6f}")

    sampler = BalancedSampler(
        ds.pos_idx,
        ds.neg_idx,
        batch_size=batch_size,
        pos_fraction=pos_frac,
        seed=0,
    )

    dl = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    print("\nIterating a few batches...")
    for bi, batch in enumerate(dl):
        x = batch["x"]
        y = batch["y"]
        lengths = batch["lengths"]
        is_pos = batch["is_pos"]
        counts = batch["count"]

        bsz = int(x.shape[0])
        pos_n = int(is_pos.sum().item())
        neg_n = bsz - pos_n

        print(
            f"[batch {bi}] "
            f"x={tuple(x.shape)} y={tuple(y.shape)} "
            f"len_min={int(lengths.min())} len_max={int(lengths.max())} "
            f"pos={pos_n} neg={neg_n} "
            f"count_mean={float(counts.mean()):.4f} count_max={float(counts.max()):.4f}"
        )

        assert x.ndim == 3 and y.ndim == 2
        assert x.shape[0] == y.shape[0] == lengths.shape[0]

        if bi + 1 >= max_batches:
            break

    print("\n06 check done.")


if __name__ == "__main__":
    main()
