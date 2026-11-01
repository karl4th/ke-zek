"""Energy-based voice activity on clean reference signals.

Used only to define scoring regions on *clean* single-speaker references (one channel per
speaker), where a simple energy threshold is reliable. It is not meant for noisy mixtures.
"""

from __future__ import annotations

import numpy as np

FRAME_S = 0.02
"""Analysis frame length (20 ms, non-overlapping)."""


def frame_energy_db(x: np.ndarray, sr: int, frame_s: float = FRAME_S) -> np.ndarray:
    """Mean power per frame in dB (relative to full scale 1.0)."""
    x = np.asarray(x, dtype=np.float64)
    n = int(round(frame_s * sr))
    n_frames = int(np.ceil(len(x) / n))
    padded = np.pad(x, (0, n_frames * n - len(x)))
    power = (padded.reshape(n_frames, n) ** 2).mean(axis=1)
    return 10 * np.log10(power + 1e-12)


def threshold_db(x: np.ndarray, sr: int, rel_db: float = -40.0, frame_s: float = FRAME_S) -> float:
    """Absolute activity threshold: ``rel_db`` below the 95th percentile frame energy."""
    return float(np.percentile(frame_energy_db(x, sr, frame_s), 95) + rel_db)


def dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    """Extend every active run by ``frames`` on both sides."""
    if frames <= 0 or not mask.any():
        return mask.copy()
    kernel = np.ones(2 * frames + 1)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0


def energy_vad(
    x: np.ndarray,
    sr: int,
    rel_db: float = -40.0,
    hangover_s: float = 0.1,
    frame_s: float = FRAME_S,
    abs_threshold_db: float | None = None,
) -> np.ndarray:
    """Boolean activity per frame.

    A frame is active when its energy exceeds ``abs_threshold_db`` or, by default,
    ``rel_db`` below the signal's 95th percentile frame energy. Active runs are extended by
    ``hangover_s`` so that word onsets and offsets count as speech.
    """
    energy = frame_energy_db(x, sr, frame_s)
    thr = abs_threshold_db if abs_threshold_db is not None else threshold_db(x, sr, rel_db, frame_s)
    return dilate(energy > thr, int(round(hangover_s / frame_s)))


def frames_to_samples(mask: np.ndarray, sr: int, n_samples: int, frame_s: float = FRAME_S):
    """Expand a per-frame mask to a per-sample mask of length ``n_samples``."""
    n = int(round(frame_s * sr))
    return np.repeat(mask, n)[:n_samples]
