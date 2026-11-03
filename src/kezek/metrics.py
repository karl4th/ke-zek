"""Scoring of target speaker extraction on two-speaker mixtures.

Inputs are sample-aligned mono signals: the estimate, the clean target and the clean
interferer (the other speaker). The mixture defaults to ``target + interferer``.

Regions are defined from energy VAD on the clean references, per 20 ms frame:

* **overlap**: both speakers active. The hard case.
* **solo**: only the target is active. The output should equal the input here.
* **absent**: only the interferer is active. The output should be silent here.

Metrics:

* ``sisdr_overlap`` (primary): SI-SDR in dB of the estimate on overlap regions.
* ``sisdri_overlap``: improvement over the unprocessed mixture on the same regions.
* ``sisdr_solo``: SI-SDR on solo regions (preservation of the target).
* ``leakage``: share of absent frames where the estimate is active, using the target's own
  activity threshold. This is how often the other voice is audible in the output.
* ``residual_db``: estimate energy on absent regions relative to the interferer energy there
  (0 dB = the interferer passes through untouched; lower is better).
* ``sisdr``: SI-SDR on the whole signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from kezek.vad import FRAME_S, energy_vad, frame_energy_db, frames_to_samples, threshold_db


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-10) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (Le Roux et al., 2019)."""
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if est.shape != ref.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {ref.shape}")
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = (est @ ref) / (ref @ ref + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((target @ target + eps) / (noise @ noise + eps)))


@dataclass
class Regions:
    overlap: np.ndarray
    solo: np.ndarray
    absent: np.ndarray
    target_threshold_db: float

    def seconds(self, frame_s: float = FRAME_S) -> dict[str, float]:
        return {
            "overlap_s": float(self.overlap.sum() * frame_s),
            "solo_s": float(self.solo.sum() * frame_s),
            "absent_s": float(self.absent.sum() * frame_s),
        }


def regions(target: np.ndarray, interferer: np.ndarray, sr: int, rel_db: float = -40.0):
    """Per-frame overlap / solo / absent masks from the clean references."""
    t_thr = threshold_db(target, sr, rel_db)
    i_thr = threshold_db(interferer, sr, rel_db)
    t = energy_vad(target, sr, abs_threshold_db=t_thr)
    i = energy_vad(interferer, sr, abs_threshold_db=i_thr)
    return Regions(overlap=t & i, solo=t & ~i, absent=~t & i, target_threshold_db=t_thr)


@dataclass
class ExtractionScores:
    sisdr_overlap: float
    sisdri_overlap: float
    sisdr_solo: float
    leakage: float
    residual_db: float
    sisdr: float
    overlap_s: float
    solo_s: float
    absent_s: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _region_sisdr(est, ref, frame_mask, sr, min_frames: int = 5) -> float:
    if frame_mask.sum() < min_frames:
        return float("nan")
    m = frames_to_samples(frame_mask, sr, len(ref))
    return si_sdr(est[m], ref[m])


def score_extraction(
    estimate: np.ndarray,
    target: np.ndarray,
    interferer: np.ndarray,
    sr: int,
    mixture: np.ndarray | None = None,
) -> ExtractionScores:
    """Score one extracted signal. All signals must have the same length and rate."""
    est = np.asarray(estimate, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    itf = np.asarray(interferer, dtype=np.float64)
    mix = tgt + itf if mixture is None else np.asarray(mixture, dtype=np.float64)
    if not (len(est) == len(tgt) == len(itf) == len(mix)):
        raise ValueError("estimate, target, interferer and mixture must have the same length")

    reg = regions(tgt, itf, sr)
    est_db = frame_energy_db(est, sr)
    n_absent = int(reg.absent.sum())
    leakage = (
        float((est_db[reg.absent] > reg.target_threshold_db).mean()) if n_absent else float("nan")
    )
    if n_absent:
        m = frames_to_samples(reg.absent, sr, len(est))
        residual_db = float(10 * np.log10((est[m] @ est[m] + 1e-12) / (itf[m] @ itf[m] + 1e-12)))
    else:
        residual_db = float("nan")

    ov_est = _region_sisdr(est, tgt, reg.overlap, sr)
    ov_mix = _region_sisdr(mix, tgt, reg.overlap, sr)
    return ExtractionScores(
        sisdr_overlap=ov_est,
        sisdri_overlap=ov_est - ov_mix,
        sisdr_solo=_region_sisdr(est, tgt, reg.solo, sr),
        leakage=leakage,
        residual_db=residual_db,
        sisdr=si_sdr(est, tgt),
        **reg.seconds(),
    )


# ---------------------------------------------------------------------------------------------
# K-T1: metrics by segment type, on 10 ms frames with known labels.
#
#   overlap       both active          SI-SDR (and SNR) of the output to the clean target
#   target_solo   only the target      SI-SDR, SNR, level of the output vs the clean target
#   other_solo    only the other       leak = E(R) / E(other's speech in the input), R = out - clean
#   both_silent   nobody               E(R) / L_ref
#   backchannel   target runs < 1 s    share with output energy >= clean target energy - 6 dB
#
# Frames near turn boundaries (``disputed``) are left out everywhere. Energies are pooled over
# windows (``aggregate``), SI-SDR / SNR are averaged per window.

SEGMENTS = ("overlap", "target_solo", "other_solo", "both_silent")
MIN_FRAMES = 10


def snr_db(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> float:
    est, ref = np.asarray(estimate, np.float64), np.asarray(reference, np.float64)
    return float(10 * np.log10((ref @ ref + eps) / ((ref - est) @ (ref - est) + eps)))


def segment_masks(act_t, act_o, disputed) -> dict[str, np.ndarray]:
    keep = ~np.asarray(disputed, bool)
    t, o = np.asarray(act_t, bool), np.asarray(act_o, bool)
    return {
        "overlap": t & o & keep,
        "target_solo": t & ~o & keep,
        "other_solo": ~t & o & keep,
        "both_silent": ~t & ~o & keep,
    }


def segment_scores(
    out, target, other, mixture, act_t, act_o, disputed, l_ref: float, frame: int = 160
) -> dict:
    """Scores of one window or one recording. Energies are returned as sums so that
    ``aggregate`` can pool them; SI-SDR / SNR are NaN when a segment is shorter than 0.1 s."""
    out, tgt, oth, mix = (np.asarray(x, np.float64) for x in (out, target, other, mixture))
    n = min(len(out), len(tgt), len(act_t) * frame)
    res = {}
    r = out - tgt
    for name, fm in segment_masks(act_t, act_o, disputed).items():
        m = np.repeat(fm, frame)[:n]
        res[f"{name}_frames"] = int(fm.sum())
        o, t, rr, ot, mx = out[:n][m], tgt[:n][m], r[:n][m], oth[:n][m], mix[:n][m]
        res[f"{name}_e_out"], res[f"{name}_e_tgt"] = float(o @ o), float(t @ t)
        res[f"{name}_e_res"], res[f"{name}_e_other"] = float(rr @ rr), float(ot @ ot)
        res[f"{name}_n"] = int(m.sum())
        ok = fm.sum() >= MIN_FRAMES
        if name in ("overlap", "target_solo"):
            res[f"{name}_sisdr"] = si_sdr(o, t) if ok else float("nan")
            res[f"{name}_snr"] = snr_db(o, t) if ok else float("nan")
        if name == "overlap":
            res["overlap_sisdr_mix"] = si_sdr(mx, t) if ok else float("nan")
    res["l_ref"] = float(l_ref)
    hits, total = backchannel_hits(out[:n], tgt[:n], act_t, act_o, disputed, frame)
    res["bc_hits"], res["bc_total"] = hits, total
    return res


def short_runs(act: np.ndarray, max_frames: int, merge_frames: int = 30) -> list[tuple[int, int]]:
    """Runs of ``act`` (gaps < ``merge_frames`` closed) shorter than ``max_frames``, not touching
    the edges of the signal (a run cut by the window edge is not a short reply)."""
    a = np.asarray(act, bool).copy()
    idx = np.flatnonzero(a)
    for i, j in zip(idx[:-1], idx[1:], strict=False):
        if 1 < j - i <= merge_frames:
            a[i:j] = True
    edges = np.flatnonzero(np.diff(np.concatenate([[0], a.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2], strict=False))
    return [(s, e) for s, e in runs if e - s < max_frames and s > 0 and e < len(a)]


def backchannel_hits(
    out, tgt, act_t, act_o, disputed, frame: int = 160, max_s: float = 1.0, drop_db: float = -6.0
):
    """Target replies shorter than 1 s said over the other's speech: how many survive."""
    hits = total = 0
    for s, e in short_runs(act_t, int(round(max_s / 0.01))):
        if not np.asarray(act_o, bool)[s:e].any():
            continue
        a, b = s * frame, e * frame
        e_out, e_t = float(out[a:b] @ out[a:b]), float(tgt[a:b] @ tgt[a:b])
        total += 1
        hits += int(e_out >= e_t * 10 ** (drop_db / 10))
    return hits, total


def _db(num: float, den: float) -> float:
    return float(10 * np.log10((num + 1e-20) / (den + 1e-20))) if den > 0 else float("nan")


def aggregate(scores: list[dict]) -> dict[str, float]:
    """Pool a list of ``segment_scores`` into the K-T1 metrics."""

    def tot(k):
        return float(sum(s[k] for s in scores))

    def mean(k):
        v = [s[k] for s in scores if k in s and np.isfinite(s[k])]
        return float(np.mean(v)) if v else float("nan")

    n_bs = tot("both_silent_n")
    l_ref_w = sum(s["l_ref"] * s["both_silent_n"] for s in scores)
    out = {
        "overlap_sisdr": mean("overlap_sisdr"),
        "overlap_sisdri": mean("overlap_sisdr") - mean("overlap_sisdr_mix"),
        "overlap_snr": mean("overlap_snr"),
        "target_solo_sisdr": mean("target_solo_sisdr"),
        "target_solo_snr": mean("target_solo_snr"),
        "target_solo_level_db": _db(tot("target_solo_e_out"), tot("target_solo_e_tgt")),
        "other_solo_leak_db": _db(tot("other_solo_e_res"), tot("other_solo_e_other")),
        "both_silent_db": _db(tot("both_silent_e_res") / max(n_bs, 1), l_ref_w / max(n_bs, 1)),
        "backchannel_recall": tot("bc_hits") / tot("bc_total") if tot("bc_total") else float("nan"),
        "backchannel_n": tot("bc_total"),
    }
    for name in SEGMENTS:
        out[f"{name}_s"] = tot(f"{name}_frames") * 0.01
    return out


def bootstrap_ci(
    scores: list[dict], n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> dict[str, list[float]]:
    """95% bootstrap interval of every ``aggregate`` metric, resampling windows or recordings."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot = [aggregate([scores[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)]
    q = (100 * (1 - level) / 2, 100 * (1 + level) / 2)
    out = {}
    for k in boot[0]:
        v = np.array([b[k] for b in boot], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = [float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))] if len(v) else []
    return out


def share_ci(flags, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> list[float]:
    """95% bootstrap interval of the mean of per-window 0/1 or real values."""
    x = np.asarray(flags, dtype=float)
    if not len(x):
        return []
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    q = (100 * (1 - level) / 2, 100 * (1 + level) / 2)
    return [float(np.percentile(m, q[0])), float(np.percentile(m, q[1]))]


def wrong_speaker(
    cos_target: np.ndarray, cos_other: np.ndarray, act_t_frames: np.ndarray, min_frames: int = 150
):
    """Share of windows where the output is closer (ECAPA cosine) to the other speaker than to
    the enrollment, over windows where the target is active >= 1.5 s.
    Returns ``(share, n_used, n_skipped)``."""
    act = np.asarray(act_t_frames)
    ok = act >= min_frames
    if not ok.any():
        return float("nan"), 0, int((~ok).sum())
    wrong = np.asarray(cos_target)[ok] < np.asarray(cos_other)[ok]
    return float(wrong.mean()), int(ok.sum()), int((~ok).sum())
