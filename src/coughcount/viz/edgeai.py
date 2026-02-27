from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from coughcount.audio.io import read_wav
from coughcount.audio.features import stft_logmag
from coughcount.data.density import make_density


def plot_cough_sample(
    wav_path: Path,
    starts: np.ndarray,
    ends: np.ndarray,
    sigma_sec: float = 0.05,
    stft_win: int = 1024,
    stft_hop: int = 256,
    max_hz: int = 8000,
    title: str | None = None,
) -> None:
    data, sr = read_wav(wav_path, dtype="float32")
    dur = len(data) / sr
    centers = (
        0.5 * (starts + ends) if starts.size > 0 else np.array([], dtype=np.float32)
    )
    f, tt, S = stft_logmag(data, sr, win=stft_win, hop=stft_hop)

    if len(tt) >= 2:
        frame_hz = 1.0 / float(tt[1] - tt[0])
    else:
        frame_hz = float(sr) / float(stft_hop)

    t_density, y_density = make_density(
        centers_sec=centers,
        frame_times=tt,
        kernel="gaussian",
        sigma_sec=sigma_sec,
    )

    fig = plt.figure(figsize=(12, 10))
    if title:
        fig.suptitle(title, fontsize=12)

    ax1 = plt.subplot(3, 1, 1)
    t = np.arange(len(data)) / float(sr)
    ax1.plot(t, data, linewidth=0.8)
    ax1.set_ylabel("Wave")
    ax1.set_xlim(0.0, dur)
    for s, e in zip(starts, ends):
        ax1.axvspan(s, e, color="red", alpha=0.3)

    ax2 = plt.subplot(3, 1, 2)
    ax2.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=[
            float(tt[0]) if len(tt) else 0.0,
            float(tt[-1]) if len(tt) else float(dur),
            float(f[0]),
            float(f[-1]),
        ],
    )
    ax2.set_ylabel("Hz")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("STFT Log-Magnitude")
    ax2.set_ylim(0, min(max_hz, sr / 2))

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t_density, y_density, linewidth=1.5)
    ax3.set_xlim(0.0, dur)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Density")
    ax3.set_title(f"Density (sigma={sigma_sec:.3f} sec) | sum={y_density.sum():.3f}")
    for c in centers:
        ax3.axvline(c, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
