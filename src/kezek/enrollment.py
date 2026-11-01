"""Pick an enrollment (voice sample) of the target speaker.

The enrollment is solo speech of the target: stretches where the target talks and the other
speaker does not. For evaluation it must come from a different part of the recording than the
segment being scored, otherwise the model is handed the answer.
"""

from __future__ import annotations

import numpy as np

from kezek.vad import FRAME_S


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e)) for s, e in zip(changes[0::2], changes[1::2], strict=True)]


def pick_enrollment(
    signal: np.ndarray,
    solo: np.ndarray,
    sr: int,
    duration_s: float = 10.0,
    exclude_s: tuple[float, float] | None = None,
    min_chunk_s: float = 0.5,
    frame_s: float = FRAME_S,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Concatenate the longest solo stretches of ``signal`` until ``duration_s`` is reached.

    ``solo`` is a per-frame mask of target-only speech. Frames inside ``exclude_s`` (seconds)
    are never used. Returns the enrollment audio and the ``(start_s, end_s)`` stretches used.
    Raises ``ValueError`` if there is less solo speech than ``duration_s``.
    """
    mask = np.asarray(solo, dtype=bool).copy()
    if exclude_s is not None:
        lo, hi = int(exclude_s[0] / frame_s), int(np.ceil(exclude_s[1] / frame_s))
        mask[max(0, lo) : max(0, hi)] = False
    min_frames = int(round(min_chunk_s / frame_s))
    runs = sorted((r for r in _runs(mask) if r[1] - r[0] >= min_frames), key=lambda r: r[0] - r[1])

    need = int(round(duration_s / frame_s))
    chosen, total = [], 0
    for s, e in runs:
        take = min(e - s, need - total)
        chosen.append((s, s + take))
        total += take
        if total >= need:
            break
    if total < need:
        raise ValueError(
            f"only {total * frame_s:.1f} s of solo speech available, need {duration_s} s"
        )
    chosen.sort()
    n = int(round(frame_s * sr))
    audio = np.concatenate([signal[s * n : e * n] for s, e in chosen])
    return audio, [(s * frame_s, e * frame_s) for s, e in chosen]
