from __future__ import annotations

import numpy as np
from scipy.signal import stft

def stft_logmag(data: np.ndarray, sr: int, win: int = 1024, hop: int = 256):
    f, t, Z = stft(
        data, fs=sr, nperseg=win, noverlap=win - hop, boundary=None, padded=False
    )
    S = np.log10(np.maximum(1e-10, np.abs(Z)))
    return f, t, S