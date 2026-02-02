from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from coughcount.data.dataset import EdgeAIWindowDataset, pad_collate
from coughcount.data.sampling import BalancedSampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--mic", type=str, default="both", choices=["out", "body", "both"])
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--hop-sec", type=float, default=4.0)
    ap.add_argument("--pos-threshold", type=float, default=0.01)

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pos-frac", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=5)
    args = ap.parse_args()

    ds = EdgeAIWindowDataset(
        split=args.split,
        mic=args.mic,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        pos_threshold=args.pos_threshold,
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
        batch_size=args.batch_size,
        pos_fraction=args.pos_frac,
        seed=0,
    )

    dl = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
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

        if bi + 1 >= args.max_batches:
            break

    print("\n06 check done.")


if __name__ == "__main__":
    main()
