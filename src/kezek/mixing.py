"""Build two-speaker training mixtures with a known answer.

A mixture is the sum of two single-speaker sources, so the clean target is known exactly.
Two ways to place the sources:

* ``mix_pair``: overlay two clips at a given offset and target-to-interferer ratio. This is the
  classic Libri2Mix-style mixture: mostly full overlap.
* ``conversation_schedule`` + ``render``: alternate turns of A and B like a real conversation,
  mostly one speaker at a time with short overlaps. Real two-person speech has ~10% overlap,
  so this matches the test condition much better.

Take both speakers' clips from the same recording where possible (same room and microphone),
so a model cannot separate by acoustics instead of voice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(10 * np.log10((x @ x) / max(len(x), 1) + 1e-12))


def set_rms_db(x: np.ndarray, db: float) -> np.ndarray:
    """Scale ``x`` to an RMS level of ``db`` dBFS."""
    x = np.asarray(x, dtype=np.float64)
    return x * 10 ** ((db - rms_db(x)) / 20)


def mix_pair(
    target: np.ndarray, interferer: np.ndarray, sir_db: float = 0.0, offset: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlay ``interferer`` on ``target`` starting ``offset`` samples later (may be negative).

    The interferer is scaled so that the target-to-interferer ratio (active RMS of the two
    clips) is ``sir_db``. Returns ``(mixture, target, interferer)``, all the same length.
    """
    tgt = np.asarray(target, dtype=np.float64)
    itf = np.asarray(interferer, dtype=np.float64)
    itf = itf * 10 ** ((rms_db(tgt) - rms_db(itf) - sir_db) / 20)
    t0, i0 = max(0, -offset), max(0, offset)
    n = max(t0 + len(tgt), i0 + len(itf))
    t_out, i_out = np.zeros(n), np.zeros(n)
    t_out[t0 : t0 + len(tgt)] = tgt
    i_out[i0 : i0 + len(itf)] = itf
    return t_out + i_out, t_out, i_out


@dataclass(frozen=True)
class Placement:
    speaker: int
    """0 = target, 1 = interferer."""
    start_s: float
    clip: int
    """Index of the clip in that speaker's clip list."""


def conversation_schedule(
    durations_a: Sequence[float],
    durations_b: Sequence[float],
    rng: np.random.Generator,
    overlap_prob: float = 0.2,
    overlap_s: tuple[float, float] = (0.2, 2.0),
    gap_s: tuple[float, float] = (0.1, 1.0),
) -> list[Placement]:
    """Alternate turns A, B, A, B, ... with gaps and occasional short overlaps.

    Each next turn starts after the previous one ends plus a gap drawn from ``gap_s``, or,
    with probability ``overlap_prob``, before it ends by an amount drawn from ``overlap_s``
    (capped at half of either turn).
    """
    placements: list[Placement] = []
    queues = [list(enumerate(durations_a)), list(enumerate(durations_b))]
    speaker, prev_end, prev_dur, t = 0, 0.0, 0.0, 0.0
    while queues[0] or queues[1]:
        if not queues[speaker]:
            speaker = 1 - speaker
        idx, dur = queues[speaker].pop(0)
        if placements:
            if rng.random() < overlap_prob:
                ov = min(rng.uniform(*overlap_s), prev_dur / 2, dur / 2)
                t = max(0.0, prev_end - ov)
            else:
                t = prev_end + rng.uniform(*gap_s)
        placements.append(Placement(speaker, t, idx))
        prev_end, prev_dur = max(prev_end, t + dur), dur
        speaker = 1 - speaker
    return placements


def render(
    placements: Sequence[Placement],
    clips_a: Sequence[np.ndarray],
    clips_b: Sequence[np.ndarray],
    sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place clips on a timeline. Returns ``(mixture, target, interferer)``."""
    clips = (clips_a, clips_b)
    n = 0
    for p in placements:
        n = max(n, int(round(p.start_s * sr)) + len(clips[p.speaker][p.clip]))
    sources = np.zeros((2, n))
    for p in placements:
        clip = np.asarray(clips[p.speaker][p.clip], dtype=np.float64)
        s = int(round(p.start_s * sr))
        sources[p.speaker, s : s + len(clip)] += clip
    return sources.sum(axis=0), sources[0], sources[1]
