"""Diarization error rate on frame grids.

Follows the NIST definition (as in md-eval and pyannote.metrics): per scored frame with
``N_ref`` reference speakers, ``N_hyp`` hypothesis speakers and ``N_correct`` correctly
mapped speakers,

    missed      = max(0, N_ref - N_hyp)
    false alarm = max(0, N_hyp - N_ref)
    confusion   = min(N_ref, N_hyp) - N_correct

and DER = (missed + false alarm + confusion) / sum(N_ref). The speaker mapping is the
one-to-one assignment that maximises total co-activity (Hungarian algorithm). Overlapping
speech is scored. A collar (seconds) excludes frames around reference boundaries; the
KezekBench default is no collar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from kezek.frames import SCORING_FRAME_RATE_HZ


def optimal_mapping(ref: np.ndarray, hyp: np.ndarray) -> dict[int, int]:
    """One-to-one map ``hyp row -> ref row`` maximising co-active frames.

    Pairs that never co-occur are left unmapped.
    """
    ref = np.asarray(ref, dtype=bool)
    hyp = np.asarray(hyp, dtype=bool)
    if ref.shape[0] == 0 or hyp.shape[0] == 0:
        return {}
    overlap = hyp.astype(np.int64) @ ref.T.astype(np.int64)  # [n_hyp, n_ref]
    rows, cols = linear_sum_assignment(overlap, maximize=True)
    return {int(h): int(r) for h, r in zip(rows, cols, strict=True) if overlap[h, r] > 0}


def pad_to_same_length(*arrays: np.ndarray) -> list[np.ndarray]:
    """Zero-pad ``[S, T]`` matrices on the time axis to a common length."""
    n = max(a.shape[1] for a in arrays)
    return [np.pad(a, ((0, 0), (0, n - a.shape[1]))) for a in arrays]


def collar_mask(ref: np.ndarray, collar_s: float, frame_rate: float) -> np.ndarray:
    """``[T]`` mask, False within ``collar_s`` seconds of any reference speaker boundary."""
    n = ref.shape[1]
    keep = np.ones(n, dtype=bool)
    if collar_s <= 0 or n == 0:
        return keep
    radius = int(round(collar_s * frame_rate))
    for row in ref:
        padded = np.concatenate([[False], row, [False]])
        for b in np.flatnonzero(padded[1:] != padded[:-1]):
            keep[max(0, b - radius) : min(n, b + radius)] = False
    return keep


@dataclass
class DERResult:
    """Error components in seconds. Results can be summed across files with ``+``."""

    total: float = 0.0
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0

    @property
    def der(self) -> float:
        if self.total == 0:
            return 0.0 if self.missed + self.false_alarm + self.confusion == 0 else float("inf")
        return (self.missed + self.false_alarm + self.confusion) / self.total

    def __add__(self, other: DERResult) -> DERResult:
        return DERResult(
            self.total + other.total,
            self.missed + other.missed,
            self.false_alarm + other.false_alarm,
            self.confusion + other.confusion,
        )

    def as_dict(self) -> dict[str, float]:
        t = self.total or float("nan")
        return {
            "der": self.der,
            "missed": self.missed / t,
            "false_alarm": self.false_alarm / t,
            "confusion": self.confusion / t,
            "total_speech_s": self.total,
        }


def diarization_error(
    ref: np.ndarray,
    hyp: np.ndarray,
    frame_rate: float = SCORING_FRAME_RATE_HZ,
    uem: np.ndarray | None = None,
    collar_s: float = 0.0,
) -> DERResult:
    """DER components for boolean ``[n_ref, T]`` and ``[n_hyp, T]`` activity matrices.

    ``uem`` is an optional ``[T]`` mask of frames to score.
    """
    ref = np.asarray(ref, dtype=bool)
    hyp = np.asarray(hyp, dtype=bool)
    ref, hyp = pad_to_same_length(ref, hyp)
    n = ref.shape[1]
    scored = collar_mask(ref, collar_s, frame_rate)
    if uem is not None:
        uem = np.asarray(uem, dtype=bool)
        scored &= np.pad(uem, (0, max(0, n - uem.shape[0])))[:n]

    mapping = optimal_mapping(ref[:, scored], hyp[:, scored])
    n_ref = ref.sum(axis=0)
    n_hyp = hyp.sum(axis=0)
    n_correct = np.zeros(n, dtype=np.int64)
    for h, r in mapping.items():
        n_correct += hyp[h] & ref[r]

    frame = 1.0 / frame_rate
    s = scored
    return DERResult(
        total=float(n_ref[s].sum()) * frame,
        missed=float(np.maximum(0, n_ref - n_hyp)[s].sum()) * frame,
        false_alarm=float(np.maximum(0, n_hyp - n_ref)[s].sum()) * frame,
        confusion=float((np.minimum(n_ref, n_hyp) - n_correct)[s].sum()) * frame,
    )
