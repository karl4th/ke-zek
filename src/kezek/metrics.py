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
