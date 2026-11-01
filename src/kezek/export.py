"""Pseudo-duplex export: turn per-speaker activity of a mono episode into two channels.

Rules (see docs/architecture.md, "Export"):

1. Exactly two main speakers. The two speakers with the most speech must together hold at
   least ``min_top2_share`` of all speech in the episode, otherwise the episode is rejected.
2. Overlapping speech (two or more speakers at once) becomes silence in both channels and is
   flagged in ``overlap``; it is not separated.
3. Frames where an excluded (third) speaker talks, or where confidence is below
   ``min_confidence``, are invalid.
4. Only contiguous runs of valid frames of at least ``min_segment_s`` are exported.

Errors in diarization turn into echo and wrong turn timing in the downstream model, so
uncertain material is dropped rather than exported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from kezek.frames import MIMI_FRAME_RATE_HZ


@dataclass
class ExportConfig:
    min_top2_share: float = 0.85
    min_confidence: float = 0.5
    min_segment_s: float = 60.0
    frame_rate: float = MIMI_FRAME_RATE_HZ


@dataclass
class PseudoDuplex:
    speakers: tuple[str, str] | None
    """Labels of channel A and channel B (A has more speech); None if rejected."""
    channels: np.ndarray
    """``[2, T]`` boolean speech activity of channel A and channel B."""
    overlap: np.ndarray
    """``[T]`` frames with overlapping speech (silenced in both channels)."""
    valid: np.ndarray
    """``[T]`` frames that may be used for training."""
    segments: list[tuple[int, int]] = field(default_factory=list)
    """Exported ``[start_frame, end_frame)`` ranges."""
    top2_share: float = 0.0
    rejected: str | None = None

    @property
    def exported_seconds(self) -> float:
        return sum(e - s for s, e in self.segments) / MIMI_FRAME_RATE_HZ


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e)) for s, e in zip(changes[0::2], changes[1::2], strict=True)]


def export_pseudo_duplex(
    activity: np.ndarray,
    speakers: Sequence[str],
    confidence: np.ndarray | None = None,
    config: ExportConfig | None = None,
) -> PseudoDuplex:
    """Build the two-channel export for one episode.

    ``activity`` is a boolean ``[num_speakers, T]`` matrix at ``config.frame_rate``;
    ``confidence`` is an optional ``[T]`` array in [0, 1].
    """
    cfg = config or ExportConfig()
    activity = np.asarray(activity, dtype=bool)
    n_spk, n = activity.shape
    if n_spk != len(speakers):
        raise ValueError("activity rows must match speakers")
    empty = PseudoDuplex(
        speakers=None,
        channels=np.zeros((2, n), dtype=bool),
        overlap=np.zeros(n, dtype=bool),
        valid=np.zeros(n, dtype=bool),
    )

    per_speaker = activity.sum(axis=1)
    total = int(per_speaker.sum())
    if n_spk < 2 or total == 0:
        empty.rejected = "fewer than two speakers"
        return empty
    order = np.argsort(-per_speaker, kind="stable")
    a, b = int(order[0]), int(order[1])
    if per_speaker[b] == 0:
        empty.rejected = "fewer than two speakers"
        return empty
    share = float(per_speaker[a] + per_speaker[b]) / total
    empty.top2_share = share
    if share < cfg.min_top2_share:
        empty.rejected = f"top-2 speech share {share:.2f} < {cfg.min_top2_share}"
        return empty

    overlap = activity.sum(axis=0) >= 2
    others = np.delete(activity, [a, b], axis=0).any(axis=0) if n_spk > 2 else np.zeros(n, bool)
    valid = ~others
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)
        if confidence.shape != (n,):
            raise ValueError(f"confidence must have shape ({n},)")
        valid &= confidence >= cfg.min_confidence

    channels = np.stack([activity[a], activity[b]]) & ~overlap & valid
    min_len = int(np.ceil(cfg.min_segment_s * cfg.frame_rate))
    segments = [(s, e) for s, e in _runs(valid) if e - s >= min_len]
    return PseudoDuplex(
        speakers=(speakers[a], speakers[b]),
        channels=channels,
        overlap=overlap & valid,
        valid=valid,
        segments=segments,
        top2_share=share,
        rejected=None if segments else "no valid segment long enough",
    )
