from __future__ import annotations

from typing import Literal, Tuple
import numpy as np

KernelName = Literal["gaussian", "skewed_gaussian", "cosine"]


def _as_1d_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def centers_from_intervals(starts, ends) -> np.ndarray:
    starts = _as_1d_float32(starts)
    ends = _as_1d_float32(ends)
    n = min(starts.size, ends.size)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return 0.5 * (starts[:n] + ends[:n])


def infer_frame_hz(frame_times: np.ndarray) -> float:
    if frame_times.size < 2:
        return 1.0
    dt = np.median(np.diff(frame_times))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid frame times for inferring frame rate.")
    return float(1.0 / dt)


def centers_to_frames(centers_sec: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    centers_sec = _as_1d_float32(centers_sec)
    frame_times = _as_1d_float32(frame_times)

    if centers_sec.size == 0 or frame_times.size == 0:
        return np.zeros(0, dtype=np.int64)

    idx = np.searchsorted(frame_times, centers_sec, side="left")
    idx = np.clip(idx, 0, frame_times.size - 1)

    left_idx = np.clip(idx - 1, 0, frame_times.size - 1)
    right_dist = np.abs(frame_times[idx] - centers_sec)
    left_dist = np.abs(frame_times[left_idx] - centers_sec)
    choose_left = left_dist <= right_dist
    idx = np.where(choose_left, left_idx, idx)

    return idx.astype(np.int64)


def _add_kernel_centered(
    density: np.ndarray, center_frame: int, base_kernel: np.ndarray, rad: int
) -> None:
    n_frames = int(density.shape[0])
    if n_frames == 0:
        return

    L = max(0, center_frame - rad)
    R = min(n_frames - 1, center_frame + rad)

    kL = L - (center_frame - rad)
    kR = kL + (R - L)

    k = base_kernel[kL : kR + 1].astype(np.float32, copy=True)
    s = float(np.sum(k))
    if s > 0.0:
        k /= s
    density[L : R + 1] += k


def make_density(
    centers_sec: np.ndarray,
    frame_times: np.ndarray,
    kernel: KernelName = "gaussian",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    centers_sec = _as_1d_float32(centers_sec)
    frame_times = _as_1d_float32(frame_times)

    n_frames = int(frame_times.size)
    density = np.zeros(n_frames, dtype=np.float32)
    if n_frames == 0 or centers_sec.size == 0:
        return frame_times, density

    frame_hz = infer_frame_hz(frame_times)
    centers_frame = centers_to_frames(centers_sec, frame_times)

    if kernel == "gaussian":
        sigma_sec = float(kwargs.get("sigma_sec", 0.05))
        sigma_f = max(1.0, sigma_sec * frame_hz)
        rad = int(np.ceil(4.0 * sigma_f))
        grid = np.arange(-rad, rad + 1, dtype=np.float32)
        base_kernel = np.exp(-0.5 * (grid / np.float32(sigma_f)) ** 2).astype(
            np.float32
        )
        for c in centers_frame:
            _add_kernel_centered(density, int(c), base_kernel, rad)
        return frame_times, density
    elif kernel == "skewed_gaussian":
        sigma_left_sec = float(kwargs.get("sigma_left_sec", 0.04))
        sigma_right_sec = float(kwargs.get("sigma_right_sec", 0.08))
        sigma_left_f = max(1.0, sigma_left_sec * frame_hz)
        sigma_right_f = max(1.0, sigma_right_sec * frame_hz)
        sigma_max = max(sigma_left_f, sigma_right_f)
        rad = int(np.ceil(4.0 * sigma_max))
        grid = np.arange(-rad, rad + 1, dtype=np.float32)
        sigma_piece = np.where(
            grid < 0, np.float32(sigma_left_f), np.float32(sigma_right_f)
        )
        base_kernel = np.exp(-0.5 * (grid / sigma_piece) ** 2).astype(np.float32)
        for c in centers_frame:
            _add_kernel_centered(density, int(c), base_kernel, rad)
        return frame_times, density
    elif kernel == "cosine":
        half_width_sec = float(kwargs.get("half_width_sec", 0.10))
        rad = max(1, int(np.ceil(half_width_sec * frame_hz)))
        grid = np.arange(-rad, rad + 1, dtype=np.float32)
        base_kernel = (0.5 + 0.5 * np.cos(np.pi * grid / np.float32(rad))).astype(
            np.float32
        )
        for c in centers_frame:
            _add_kernel_centered(density, int(c), base_kernel, rad)
        return frame_times, density

    raise ValueError(f"Unsupported kernel type: {kernel}")
