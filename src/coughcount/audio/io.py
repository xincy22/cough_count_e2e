from __future__ import annotations

from pathlib import Path
import soundfile as sf


def read_wav(path: Path, dtype: str = "float32"):
    data, sr = sf.read(str(path), dtype=dtype)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sr)

def wav_info(path: Path) -> tuple[int, float]:
    info = sf.info(str(path))
    sr = int(info.samplerate)
    duration = float(info.frames) / sr
    return sr, duration