from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coughcount.paths import ProjectPaths as P
from coughcount.data.edgeai import ensure_edgeai_downloaded
from coughcount.viz.edgeai import plot_cough_sample


def _parse_list(s: str) -> np.ndarray:
    if not isinstance(s, str) or not s:
        return np.array([], dtype=np.float32)
    return np.array(json.loads(s), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index", type=int, default=-1, help=">=0 pick cough row index, -1 random"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--mic", type=str, default="out", choices=["out", "body"])
    args = ap.parse_args()

    public_root = ensure_edgeai_downloaded(P.edgeai_raw)
    df = pd.read_csv(P.edgeai_manifest_csv)
    cough = df[df["class"] == "cough"].reset_index(drop=True)
    if len(cough) == 0:
        raise RuntimeError("No cough samples found in manifest")

    rng = np.random.default_rng(args.seed)
    row = (
        cough.iloc[args.index]
        if args.index >= 0
        else cough.iloc[int(rng.integers(0, len(cough)))]
    )

    wav_rel = row["out_wav"] if args.mic == "out" else row["body_wav"]
    if not isinstance(wav_rel, str) or not wav_rel:
        raise RuntimeError(f"Invalid wav path in row: {wav_rel}")

    wav_path = public_root / wav_rel
    starts = _parse_list(row["starts"])
    ends = _parse_list(row["ends"])

    title = (
        f"{row['subject_id']} | {row['trial']} | {row['movement']} | {row['background']} | mic={args.mic}\n"
        f"{wav_rel} | sr={row['sample_rate']} | dur={row['duration']:.2f}s | events={len(starts)}"
    )
    plot_cough_sample(wav_path, starts, ends, sigma_sec=float(args.sigma), title=title)


if __name__ == "__main__":
    main()
