"""
Density图预计算脚本 - 使用最佳density核配置
从experiment.yaml读取density配置，生成密度图
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from coughcount.audio.features import stft_logmag
from coughcount.audio.io import read_wav
from coughcount.data.density import centers_from_intervals, make_density
from coughcount.paths import ProjectPaths as P


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _parse_list(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value:
        return np.array([], dtype=np.float32)
    return np.array(json.loads(value), dtype=np.float32)


def precompute_best_kernel(
    density_cfg: dict,
    output_dir: Path,
    mic: str = "both",
    stft_win: int = 1024,
    stft_hop: int = 256,
) -> Path:
    """使用最佳核函数配置预计算密度图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel = density_cfg["kernel"]
    sigma_sec = density_cfg.get("sigma_sec", 0.05)
    sigma_left_sec = density_cfg.get("sigma_left_sec", 0.04)
    sigma_right_sec = density_cfg.get("sigma_right_sec", 0.08)

    print(f"=== Best Density Kernel ===")
    print(f"Kernel: {kernel}")
    if kernel == "gaussian":
        print(f"  sigma_sec: {sigma_sec}")
    elif kernel == "skewed_gaussian":
        print(f"  sigma_left_sec: {sigma_left_sec}")
        print(f"  sigma_right_sec: {sigma_right_sec}")
    elif kernel == "cosine":
        print(f"  half_width_sec: {sigma_sec}")
    print(f"Output: {output_dir}\n")

    df = pd.read_csv(P.edgeai_manifest_csv)
    public_root = P.edgeai_raw / "public_dataset"

    mics = ["out", "body"] if mic == "both" else [mic]

    for mic_name in mics:
        wav_col = "out_wav" if mic_name == "out" else "body_wav"
        df_mic = df[df[wav_col].astype(str).str.len() > 0].reset_index(drop=True)

        mic_dir = output_dir / mic_name
        mic_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(
            df_mic.iterrows(), total=len(df_mic), desc=f"mic={mic_name}", unit="file"
        ):
            wav_rel = str(row[wav_col])
            wav_path = public_root / wav_rel

            data, sr = read_wav(wav_path, dtype="float32")
            _, tt, S = stft_logmag(data, sr, win=stft_win, hop=stft_hop)

            starts = _parse_list(row["starts"])
            ends = _parse_list(row["ends"])
            centers = centers_from_intervals(starts, ends)

            # Build density kwargs
            density_kwargs = {"kernel": kernel}
            if kernel == "gaussian":
                density_kwargs["sigma_sec"] = sigma_sec
            elif kernel == "skewed_gaussian":
                density_kwargs["sigma_left_sec"] = sigma_left_sec
                density_kwargs["sigma_right_sec"] = sigma_right_sec
            elif kernel == "cosine":
                density_kwargs["half_width_sec"] = sigma_sec

            _, density = make_density(
                centers_sec=centers,
                frame_times=tt,
                **density_kwargs,
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
                "mic": mic_name,
                "stft_win": int(stft_win),
                "stft_hop": int(stft_hop),
                "kernel": str(kernel),
                "sigma_sec": float(sigma_sec),
            }
            if kernel == "skewed_gaussian":
                meta["sigma_left_sec"] = float(sigma_left_sec)
                meta["sigma_right_sec"] = float(sigma_right_sec)

            with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)

        print(f"Done mic={mic_name}, saved to {mic_dir}")

    # Copy splits.json
    splits_src = P.edgeai_splits_json
    splits_dst = output_dir / "splits.json"
    if splits_src.exists() and not splits_dst.exists():
        import shutil
        shutil.copy(splits_src, splits_dst)

    print(f"\nComplete: {output_dir}")
    return output_dir


def main() -> None:
    cfg = load_config()
    density_cfg = cfg["density"]

    # Use experiment data directory
    exp_dir = Path(__file__).parent.parent
    output_dir = exp_dir / "data"

    print(f"Output directory: {output_dir}\n")

    precompute_best_kernel(
        density_cfg,
        output_dir=output_dir,
        mic="both",
        stft_win=1024,
        stft_hop=256,
    )

    print("Density map precomputation complete!")


if __name__ == "__main__":
    main()
