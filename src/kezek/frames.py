"""Frame grid conventions.

Kezek works on the Mimi frame grid: 12.5 frames per second (80 ms per frame), so its output
lines up with the frames of downstream speech models that use Mimi without resampling.

Frame ``i`` covers the half-open interval ``[i / rate, (i + 1) / rate)``. A segment marks frame
``i`` active when the frame centre ``(i + 0.5) / rate`` falls inside the segment. For scoring,
annotations are rasterised on a finer grid (``SCORING_FRAME_RATE_HZ``, 10 ms), which keeps the
discretisation error far below the boundary tolerances we report.
"""

from __future__ import annotations

import math

import numpy as np

MIMI_FRAME_RATE_HZ: float = 12.5
"""Output frame rate of Kezek (Mimi frames)."""

SCORING_FRAME_RATE_HZ: float = 100.0
"""Frame rate used to rasterise reference and hypothesis annotations for scoring."""


def num_frames(duration_s: float, frame_rate: float = MIMI_FRAME_RATE_HZ) -> int:
    """Number of frames needed to cover ``duration_s`` seconds."""
    if duration_s < 0:
        raise ValueError(f"duration must be non-negative, got {duration_s}")
    return int(math.ceil(round(duration_s * frame_rate, 9)))


def frame_centers(n: int, frame_rate: float = MIMI_FRAME_RATE_HZ) -> np.ndarray:
    """Centre time (seconds) of each of ``n`` frames."""
    return (np.arange(n) + 0.5) / frame_rate


def time_to_frame(t: float, frame_rate: float = MIMI_FRAME_RATE_HZ) -> int:
    """Index of the frame that contains time ``t``."""
    return int(math.floor(round(t * frame_rate, 9)))


def frame_to_time(i: int, frame_rate: float = MIMI_FRAME_RATE_HZ) -> float:
    """Start time (seconds) of frame ``i``."""
    return i / frame_rate


def interval_to_mask(
    start: float, end: float, n: int, frame_rate: float = MIMI_FRAME_RATE_HZ
) -> np.ndarray:
    """Boolean mask of the frames whose centre lies in ``[start, end)``."""
    centers = frame_centers(n, frame_rate)
    return (centers >= start) & (centers < end)
