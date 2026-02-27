from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tqdm import tqdm

from coughcount.audio.io import wav_info
from coughcount.utils.checksum import md5_file

ZENODO_URL = "https://zenodo.org/records/7562332/files/public_dataset.zip?download=1"
EXPECTED_MD5 = "37419b515b29ed6115bcb9ac422eb4a3"


def download_file(url: str, out_path: Path, chunk_size: int = 256 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resume_pos = out_path.stat().st_size if out_path.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_size = int(total) + resume_pos if total is not None else None

        mode = "ab" if resume_pos > 0 else "wb"
        with out_path.open(mode) as f, tqdm(
            total=total_size,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {out_path.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _safe_extract(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            target = (extract_to / member.filename).resolve()
            if not str(target).startswith(str(extract_to.resolve())):
                raise RuntimeError(
                    f"Unsafe path detected in zip file: {member.filename}"
                )
        z.extractall(extract_to)


def ensure_edgeai_downloaded(edgeai_raw_dir: Path) -> Path:
    edgeai_raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = edgeai_raw_dir / "public_dataset.zip"
    if not zip_path.exists():
        download_file(ZENODO_URL, zip_path)

    got = md5_file(zip_path)
    if got.lower() != EXPECTED_MD5.lower():
        raise RuntimeError(
            f"MD5 mismatch: expected {EXPECTED_MD5}, got {got}\n"
            f"you may need to delete {zip_path} and try again."
        )

    public_dataset = edgeai_raw_dir / "public_dataset"
    if not public_dataset.exists():
        _safe_extract(zip_path, edgeai_raw_dir)
    return public_dataset


def parse_components(public_dataset_root: Path, out_wav_path: Path) -> dict[str, str]:
    rel = out_wav_path.relative_to(public_dataset_root)
    parts = rel.parts
    if len(parts) < 6:
        raise ValueError(f"Unexpected path structure: {out_wav_path}")

    return {
        "subject_id": parts[0],
        "trial": parts[1],
        "movement": parts[2],
        "background": parts[3],
        "class": parts[4],
    }


def build_manifest(public_dataset_root: Path, manifest_csv: Path) -> pd.DataFrame:
    wavs = list(public_dataset_root.rglob("outward_facing_mic.wav"))

    if not wavs:
        raise RuntimeError(
            f"No outward_facing_mic.wav files found under {public_dataset_root}"
        )

    rows: list[dict[str, Any]] = []
    for out_wav in tqdm(wavs, desc="Scanning EdgeAI segments"):
        seg_dir = out_wav.parent
        body_wav = seg_dir / "body_facing_mic.wav"
        imu_csv = seg_dir / "imu.csv"
        gt_json = seg_dir / "ground_truth.json"

        meta = parse_components(public_dataset_root, out_wav)
        sr, dur = wav_info(out_wav)

        starts: list[float] = []
        ends: list[float] = []
        cough_count = 0

        if meta["class"] == "cough":
            if not gt_json.exists():
                raise FileNotFoundError(f"Ground truth JSON not found: {gt_json}")
            with gt_json.open("r") as f:
                gt_data = json.load(f)
            try:
                starts = [float(x) for x in gt_data["start_times"]]
                ends = [float(x) for x in gt_data["end_times"]]
                cough_count = len(starts)
            except KeyError as e:
                raise KeyError(
                    f"Missing expected keys in ground truth JSON: {e}"
                ) from e

            if len(starts) != len(ends):
                raise ValueError(f"Mismatched start/end times in {gt_json}")

        rows.append(
            {
                **meta,
                "out_wav": str(out_wav.relative_to(public_dataset_root)),
                "body_wav": (
                    str(body_wav.relative_to(public_dataset_root))
                    if body_wav.exists()
                    else ""
                ),
                "imu_csv": (
                    str(imu_csv.relative_to(public_dataset_root))
                    if imu_csv.exists()
                    else ""
                ),
                "ground_truth_json": (
                    str(gt_json.relative_to(public_dataset_root))
                    if gt_json.exists()
                    else ""
                ),
                "sample_rate": sr,
                "duration": dur,
                "cough_count": cough_count,
                "starts": json.dumps(starts),
                "ends": json.dumps(ends),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["subject_id", "trial", "movement", "background", "class"]
    )
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_csv, index=False)
    return df
