from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from coughcount.audio.features import stft_logmag
from coughcount.audio.io import read_wav
from coughcount.data.density import centers_from_intervals, make_density
from coughcount.paths import ProjectPaths as P


def _parse_list(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value:
        return np.array([], dtype=np.float32)
    return np.array(json.loads(value), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", type=str, default="both", choices=["out", "body", "both"])
    ap.add_argument("--out-dir", type=Path, default=P.edgeai_npy)
    ap.add_argument("--stft-win", type=int, default=1024)
    ap.add_argument("--stft-hop", type=int, default=256)
    ap.add_argument("--kernel", type=str, default="gaussian")
    ap.add_argument("--sigma-sec", type=float, default=0.05)
    args = ap.parse_args()

    out_dir: Path = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(P.edgeai_manifest_csv)
    public_root = P.edgeai_raw / "public_dataset"

    mics = ["out", "body"] if args.mic == "both" else [args.mic]

    for mic in mics:
        wav_col = "out_wav" if mic == "out" else "body_wav"
        df_mic = df[df[wav_col].astype(str).str.len() > 0].reset_index(drop=True)

        mic_dir = out_dir / mic
        mic_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(
            df_mic.iterrows(), total=len(df_mic), desc=f"mic={mic}", unit="file"
        ):
            wav_rel = str(row[wav_col])
            wav_path = public_root / wav_rel

            data, sr = read_wav(wav_path, dtype="float32")
            _, tt, S = stft_logmag(data, sr, win=args.stft_win, hop=args.stft_hop)

            starts = _parse_list(row["starts"])
            ends = _parse_list(row["ends"])
            centers = centers_from_intervals(starts, ends)

            _, density = make_density(
                centers_sec=centers,
                frame_times=tt,
                kernel=args.kernel,
                sigma_sec=args.sigma_sec,
            )

            stem = wav_rel.replace("\\", "_").replace("/", "_").replace(".wav", "")
            sample_dir = mic_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            np.save(sample_dir / "S.npy", S.astype(np.float32, copy=False))
            np.save(sample_dir / "t.npy", tt.astype(np.float32, copy=False))
            np.save(sample_dir / "density.npy", density.astype(np.float32, copy=False))

            meta = {
                "sr": int(sr),
                "wav_rel": wav_rel,
                "subject_id": str(row["subject_id"]),
                "trial": str(row["trial"]),
                "movement": str(row["movement"]),
                "background": str(row["background"]),
                "class": str(row["class"]),
                "mic": mic,
                "stft_win": int(args.stft_win),
                "stft_hop": int(args.stft_hop),
                "kernel": str(args.kernel),
                "sigma_sec": float(args.sigma_sec),
            }
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)

        print(f"Done mic={mic}, saved to {mic_dir}")


if __name__ == "__main__":
    main()
