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


# ---------------------------------------------------------------------------------------------
# Conversation simulator (K-T1): a 4 s crop of a two-person conversation with frame labels.
#
# Turns are rendered whole through the linear part of the domain chain (reverb, band), then the
# crop is cut, then the additive and non-linear part (noise, music, gain, clipping, codec) is
# applied to the summed crop only. The clean target is the target after the linear part, so it
# carries the reverb tail; labels come from the dry turns and do not cover that tail.

FRAME_S_CONV = 0.01
"""Label frame for the simulator, the K-T1 loss and the K-T1 metrics (10 ms)."""

LAYOUTS = ("turns", "short_overlaps", "full_overlap")


@dataclass(frozen=True)
class ConversationConfig:
    sr: int = 16000
    crop_s: float = 4.0
    layout_p: tuple[float, float, float] = (0.60, 0.25, 0.15)
    """Window layouts: turns with pauses, turns with short overlaps, full overlap."""
    pause_s: tuple[float, float] = (0.2, 3.0)
    overlap_s: tuple[float, float] = (0.2, 2.0)
    turn_s: tuple[float, float] = (1.0, 5.0)
    sir_db: tuple[float, float] = (-5.0, 5.0)
    """Target-to-other ratio over each speaker's active frames."""
    backchannel_p: float = 0.2
    backchannel_s: tuple[float, float] = (0.3, 1.0)
    enroll_s: tuple[float, float] = (6.0, 10.0)
    disputed_s: float = 0.15
    """Frames this close to a turn boundary are excluded from metrics and the silence loss."""
    peak: float = 0.5
    """Peak of the summed crop before gain; leaves 6 dB of headroom for the gain augmentation."""


@dataclass
class Episode:
    mixture: np.ndarray
    target: np.ndarray
    """Clean target: target speech after reverb, band and gain, without noise, music, codec."""
    other: np.ndarray
    """The other speaker as it is in the mixture (same chain), without noise."""
    act_t: np.ndarray
    act_o: np.ndarray
    disputed: np.ndarray
    l_ref: float
    """Mean energy of the clean target over its active frames in the whole rendered turns."""
    enrollment: np.ndarray
    enrollment_other: np.ndarray
    layout: int
    backchannel: bool
    target_first: bool
    sir_db: float
    clean: bool
    """No domain augmentation in this window."""
    full: dict | None = None
    """Whole-timeline tracks and labels, for checks."""


def frame_count(n_samples: int, sr: int, frame_s: float = FRAME_S_CONV) -> int:
    return n_samples // int(round(frame_s * sr))


def active_frames(x: np.ndarray, sr: int, frame_s: float = FRAME_S_CONV) -> np.ndarray:
    """Energy VAD (40 dB below the 95th percentile frame, 100 ms hangover) on a dry track."""
    from kezek.vad import energy_vad

    n = int(round(frame_s * sr))
    act = energy_vad(x[: len(x) // n * n], sr, frame_s=frame_s)
    return act[: len(x) // n]


def reference_level(clean: np.ndarray, act: np.ndarray, enrollment: np.ndarray, sr: int) -> float:
    """L_ref: mean energy of ``clean`` on its active frames; of the enrollment if none."""
    n = int(round(FRAME_S_CONV * sr))
    if act.any():
        m = np.repeat(act, n)[: len(clean)]
        return float(np.mean(np.asarray(clean[: len(m)], dtype=np.float64)[m] ** 2))
    return float(np.mean(np.asarray(enrollment, dtype=np.float64) ** 2)) + 1e-12


def _chunk(utts: Sequence[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    u = utts[int(rng.integers(len(utts)))]
    if len(u) <= n:
        return np.asarray(u, dtype=np.float64)
    s = int(rng.integers(0, len(u) - n + 1))
    return np.asarray(u[s : s + n], dtype=np.float64)


def _voiced_chunk(utts, n, rng, sr, tries: int = 20) -> np.ndarray:
    """A short chunk that is mostly speech (for backchannels)."""
    best, best_share = None, -1.0
    for _ in range(tries):
        c = _chunk(utts, n, rng)
        share = float(active_frames(c, sr).mean()) if len(c) >= sr // 10 else 0.0
        if share > best_share:
            best, best_share = c, share
        if share >= 0.7:
            break
    return best


def simulate_conversation(
    utts_t: Sequence[np.ndarray],
    utts_o: Sequence[np.ndarray],
    enr_t: np.ndarray,
    enr_o: np.ndarray,
    rng: np.random.Generator,
    chain,
    cfg: ConversationConfig | None = None,
    keep_full: bool = False,
) -> Episode:
    """One training or validation window.

    ``utts_t`` / ``utts_o``: dry utterances of the target / other speaker to cut turns from.
    ``enr_t`` / ``enr_o``: other dry utterances of the same speakers, for the enrollments.
    ``chain``: a ``kezek.augment.Chain`` (one per window), the "channel" of this recording.
    """
    cfg = cfg or ConversationConfig()
    sr = cfg.sr
    fs = int(round(FRAME_S_CONV * sr))
    crop = frame_count(int(cfg.crop_s * sr), sr) * fs
    layout = int(rng.choice(3, p=cfg.layout_p))
    target_first = bool(rng.random() < 0.5)
    sir = float(rng.uniform(*cfg.sir_db))
    backchannel = layout != 2 and bool(rng.random() < cfg.backchannel_p)
    utts = (utts_t, utts_o)

    segs: list[tuple[int, int, np.ndarray]] = []  # (speaker, start sample, dry audio)
    if layout == 2:
        pad = sr
        segs = [
            (0, 0, _chunk(utts_t, crop + 2 * pad, rng)),
            (1, 0, _chunk(utts_o, crop + 2 * pad, rng)),
        ]
        crop_start = pad
    else:
        spk = 0 if target_first else 1
        first = _chunk(utts[spk], int(rng.uniform(*cfg.turn_s) * sr), rng)
        segs.append((spk, 0, first))
        if backchannel and spk == 0:
            # the other's turn must reach into the crop with room for the backchannel
            crop_start = max(0, len(first) - int(rng.uniform(0.3, 1.5) * sr))
        else:
            crop_start = int(rng.uniform(0, max(1, len(first) - 0.2 * sr)))
        crop_start = crop_start // fs * fs
        prev_end, prev_len = len(first), len(first)
        while prev_end < crop_start + crop:
            spk = 1 - spk
            x = _chunk(utts[spk], int(rng.uniform(*cfg.turn_s) * sr), rng)
            if layout == 1:
                ov = min(rng.uniform(*cfg.overlap_s) * sr, prev_len / 2, len(x) / 2)
                start = int(prev_end - ov)
            else:
                hi = 1.0 if backchannel and len(segs) == 1 and segs[0][0] == 0 else cfg.pause_s[1]
                start = int(prev_end + rng.uniform(cfg.pause_s[0], max(cfg.pause_s[0], hi)) * sr)
            segs.append((spk, start, x))
            prev_end, prev_len = max(prev_end, start + len(x)), len(x)
        if backchannel:
            n_bc = int(rng.uniform(*cfg.backchannel_s) * sr)
            room = []
            for s, st, x in segs:
                a, b = max(st, crop_start), min(st + len(x), crop_start + crop)
                if s == 1 and b - a >= n_bc + int(0.4 * sr):
                    room.append((a, b))
            if room:
                a, b = room[int(rng.integers(len(room)))]
                st = int(rng.integers(a + int(0.2 * sr), b - n_bc - int(0.2 * sr) + 1))
                segs.append((0, st, _voiced_chunk(utts_t, n_bc, rng, sr)))
            else:
                backchannel = False

    total = max(st + len(x) for _, st, x in segs) + sr  # room for a reverb tail
    total = -(-total // fs) * fs
    dry = np.zeros((2, total))
    in_turn = np.zeros((2, total // fs), dtype=bool)
    edges = np.zeros(total // fs, dtype=bool)
    for s, st, x in segs:
        dry[s, st : st + len(x)] += x
        a, b = st // fs, (st + len(x)) // fs
        in_turn[s, a:b] = True
        w = int(round(cfg.disputed_s / FRAME_S_CONV))
        for e in (a, b):
            edges[max(0, e - w) : e + w + 1] = True
    act = np.stack([active_frames(dry[0], sr) & in_turn[0], active_frames(dry[1], sr) & in_turn[1]])

    wet = np.stack([chain.sources(dry[0]), chain.sources(dry[1])])
    m_t, m_o = np.repeat(act[0], fs), np.repeat(act[1], fs)
    e_t = max(float(np.mean(wet[0][m_t] ** 2)) if m_t.any() else 0.0, 1e-12)
    e_o = max(float(np.mean(wet[1][m_o] ** 2)) if m_o.any() else 0.0, 1e-12)
    wet[1] *= np.sqrt(e_t / e_o / 10 ** (sir / 10))
    l_ref = e_t

    sl = slice(crop_start, crop_start + crop)
    fsl = slice(crop_start // fs, (crop_start + crop) // fs)
    t, o = wet[0][sl].copy(), wet[1][sl].copy()
    g0 = cfg.peak / max(np.abs(t + o).max(), 1e-9)
    t, o, l_ref = t * g0, o * g0, l_ref * g0**2
    a_t, a_o = act[0][fsl], act[1][fsl]
    any_act = np.repeat(a_t | a_o, fs)
    speech_e = (
        float(np.mean((t + o)[any_act] ** 2)) if any_act.any() else float(np.mean((t + o) ** 2))
    )
    mix, g = chain.finish(t + o, speech_e, rng)
    t, o, l_ref = t * g, o * g, l_ref * g**2

    def enroll(u):
        n = int(rng.uniform(*cfg.enroll_s) * sr)
        e = chain.sources(_chunk([u], n, rng))
        e = e * (cfg.peak / max(np.abs(e).max(), 1e-9))
        return chain.finish(e, float(np.mean(e**2)), rng)[0]

    full = None
    if keep_full:
        full = {
            "target": wet[0],
            "other": wet[1],
            "act_t": act[0],
            "act_o": act[1],
            "dry_t": dry[0],
            "dry_o": dry[1],
            "crop_start": crop_start,
            "segs": segs,
        }
    return Episode(
        mixture=np.float32(mix),
        target=np.float32(t),
        other=np.float32(o),
        act_t=a_t,
        act_o=a_o,
        disputed=edges[fsl],
        l_ref=float(l_ref),
        enrollment=np.float32(enroll(enr_t)),
        enrollment_other=np.float32(enroll(enr_o)),
        layout=layout,
        backchannel=backchannel,
        target_first=target_first,
        sir_db=sir,
        clean=chain.clean,
        full=full,
    )
