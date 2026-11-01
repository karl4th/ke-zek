"""Duplex metrics: what matters when diarization output is split into per-speaker channels.

Standard DER does not say how a pseudo-duplex (two-channel) training set built from the
diarization will look. These metrics do:

* **Boundary error** - for every reference turn onset/offset, the distance to the nearest
  onset/offset of the mapped hypothesis speaker. Reported as median and 90th percentile (ms)
  and the share of boundaries missed entirely (no hypothesis boundary within ``max_error_s``).
  Pauses shorter than ``merge_gap_s`` inside a speaker's speech are bridged first, so
  boundaries are turn boundaries rather than word gaps.
* **Leakage** - share of hypothesis speech frames, in any hypothesis speaker's channel, where
  the reference says *another* speaker is talking and the mapped speaker is silent. Leakage
  becomes echo of the partner in a speaker's channel.
* **Overlap precision / recall** - frames with two or more active speakers.
* **VAD precision / recall** - frames with any active speaker.

All inputs are boolean ``[speakers, frames]`` matrices on the same grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kezek.frames import SCORING_FRAME_RATE_HZ
from kezek.metrics.der import optimal_mapping, pad_to_same_length


def bridge_gaps(row: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill inactive runs of at most ``max_gap_frames`` that lie between two active runs."""
    row = np.asarray(row, dtype=bool).copy()
    if max_gap_frames <= 0 or not row.any():
        return row
    padded = np.concatenate([[True], row, [True]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    # changes come in pairs (gap_start, gap_end) for inactive runs
    for start, end in zip(changes[0::2], changes[1::2], strict=True):
        interior = start > 0 and end < len(row)
        if interior and end - start <= max_gap_frames:
            row[start:end] = True
    return row


def boundaries(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onset and offset frame indices of active runs."""
    padded = np.concatenate([[False], np.asarray(row, dtype=bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return changes[0::2], changes[1::2]


def _nearest(targets: np.ndarray, candidates: np.ndarray, cap: float) -> np.ndarray:
    if len(targets) == 0:
        return np.zeros(0)
    if len(candidates) == 0:
        return np.full(len(targets), cap, dtype=float)
    d = np.abs(targets[:, None] - candidates[None, :]).min(axis=1).astype(float)
    return np.minimum(d, cap)


@dataclass
class DuplexCounts:
    """Additive counts; sum across files with ``+`` and call ``summary()``."""

    boundary_errors_s: list[float] = field(default_factory=list)
    max_error_s: float = 1.0
    hyp_speech_frames: int = 0
    leaked_frames: int = 0
    overlap_tp: int = 0
    overlap_fp: int = 0
    overlap_fn: int = 0
    vad_tp: int = 0
    vad_fp: int = 0
    vad_fn: int = 0

    def __add__(self, o: DuplexCounts) -> DuplexCounts:
        return DuplexCounts(
            self.boundary_errors_s + o.boundary_errors_s,
            max(self.max_error_s, o.max_error_s),
            self.hyp_speech_frames + o.hyp_speech_frames,
            self.leaked_frames + o.leaked_frames,
            self.overlap_tp + o.overlap_tp,
            self.overlap_fp + o.overlap_fp,
            self.overlap_fn + o.overlap_fn,
            self.vad_tp + o.vad_tp,
            self.vad_fp + o.vad_fp,
            self.vad_fn + o.vad_fn,
        )

    def summary(self) -> dict[str, float]:
        def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
            p = tp / (tp + fp) if tp + fp else float("nan")
            r = tp / (tp + fn) if tp + fn else float("nan")
            f = 2 * p * r / (p + r) if p + r and not np.isnan(p + r) else float("nan")
            return p, r, f

        errs = np.asarray(self.boundary_errors_s, dtype=float)
        has = len(errs) > 0
        op, orc, of = prf(self.overlap_tp, self.overlap_fp, self.overlap_fn)
        vp, vr, vf = prf(self.vad_tp, self.vad_fp, self.vad_fn)
        return {
            "boundary_median_ms": float(np.median(errs) * 1000) if has else float("nan"),
            "boundary_p90_ms": float(np.percentile(errs, 90) * 1000) if has else float("nan"),
            "boundary_missed": float((errs >= self.max_error_s).mean()) if has else float("nan"),
            "num_boundaries": float(len(errs)),
            "leakage": (
                self.leaked_frames / self.hyp_speech_frames
                if self.hyp_speech_frames
                else float("nan")
            ),
            "overlap_precision": op,
            "overlap_recall": orc,
            "overlap_f1": of,
            "vad_precision": vp,
            "vad_recall": vr,
            "vad_f1": vf,
        }


def duplex_counts(
    ref: np.ndarray,
    hyp: np.ndarray,
    frame_rate: float = SCORING_FRAME_RATE_HZ,
    merge_gap_s: float = 0.25,
    max_error_s: float = 1.0,
    uem: np.ndarray | None = None,
) -> DuplexCounts:
    """Duplex metric counts for boolean ``[n_ref, T]`` and ``[n_hyp, T]`` matrices."""
    ref, hyp = pad_to_same_length(np.asarray(ref, bool), np.asarray(hyp, bool))
    n = ref.shape[1]
    scored = np.ones(n, dtype=bool)
    if uem is not None:
        uem = np.asarray(uem, dtype=bool)
        scored = np.pad(uem, (0, max(0, n - uem.shape[0])))[:n]
    ref = ref & scored
    hyp = hyp & scored

    mapping = optimal_mapping(ref, hyp)
    ref_of_hyp = mapping
    hyp_of_ref = {r: h for h, r in mapping.items()}
    gap = int(round(merge_gap_s * frame_rate))

    # Boundary errors, per reference speaker, onsets matched to onsets, offsets to offsets.
    errors: list[float] = []
    for r in range(ref.shape[0]):
        r_on, r_off = boundaries(bridge_gaps(ref[r], gap))
        if r in hyp_of_ref:
            h_on, h_off = boundaries(bridge_gaps(hyp[hyp_of_ref[r]], gap))
        else:
            h_on = h_off = np.zeros(0, dtype=int)
        for targets, cands in ((r_on, h_on), (r_off, h_off)):
            d = _nearest(targets.astype(float), cands.astype(float), max_error_s * frame_rate)
            errors.extend((d / frame_rate).tolist())

    # Leakage.
    ref_any = ref.any(axis=0)
    leaked = 0
    for h in range(hyp.shape[0]):
        own = ref[ref_of_hyp[h]] if h in ref_of_hyp else np.zeros(n, dtype=bool)
        others = ref_any if h not in ref_of_hyp else (ref.sum(axis=0) - own) > 0
        leaked += int((hyp[h] & others & ~own).sum())

    ref_ov = ref.sum(axis=0) >= 2
    hyp_ov = hyp.sum(axis=0) >= 2
    hyp_any = hyp.any(axis=0)
    return DuplexCounts(
        boundary_errors_s=errors,
        max_error_s=max_error_s,
        hyp_speech_frames=int(hyp.sum()),
        leaked_frames=leaked,
        overlap_tp=int((ref_ov & hyp_ov).sum()),
        overlap_fp=int((~ref_ov & hyp_ov & scored).sum()),
        overlap_fn=int((ref_ov & ~hyp_ov).sum()),
        vad_tp=int((ref_any & hyp_any).sum()),
        vad_fp=int((~ref_any & hyp_any & scored).sum()),
        vad_fn=int((ref_any & ~hyp_any).sum()),
    )
