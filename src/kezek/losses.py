"""K-T1 loss: conversation behaviour with a fixed output scale.

    loss = -SNR_tau(active target frames) + 0.3 * silence penalty + 0.1 * speaker term

* SNR_tau = 10 log10(|s|^2 / (|s - s_hat|^2 + tau |s|^2)), tau = 1e-3 (ceiling 30 dB), on the
  frames where the target speaks. Unlike a scale-invariant metric it fixes the output level:
  the silence penalty, leakage and backchannel checks all depend on that level, and a
  scale-free term would let the model buy a lower penalty by turning everything down.
* Silence penalty: max(0, 10 log10(E(R) / L_ref) + 30) on the frames where the target is silent
  (turn boundaries excluded), with R = s_hat - s. Using the residual, not the output, means the
  target's own reverb tail after a turn is not punished.
* Speaker term: 1 - cos(ECAPA(s_hat on active frames), d).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FRAME = 160
"""10 ms at 16 kHz, the simulator's label frame."""


def frames_to_samples(mask: torch.Tensor, n_samples: int, frame: int = FRAME) -> torch.Tensor:
    """[B, F] frame mask -> [B, n_samples] float weights."""
    w = mask.to(torch.float32).repeat_interleave(frame, dim=-1)
    if w.shape[-1] < n_samples:
        w = F.pad(w, (0, n_samples - w.shape[-1]))
    return w[..., :n_samples]


def snr_tau(
    est: torch.Tensor, ref: torch.Tensor, w: torch.Tensor, tau: float = 1e-3
) -> torch.Tensor:
    """SNR with a soft ceiling of -10 log10(tau) dB, over the samples where ``w`` = 1. [B]"""
    s2 = (w * ref**2).sum(-1).clamp_min(1e-12)
    e2 = (w * (ref - est) ** 2).sum(-1)
    return 10 * torch.log10(s2 / (e2 + tau * s2))


def residual_db(
    est: torch.Tensor, ref: torch.Tensor, w: torch.Tensor, l_ref: torch.Tensor
) -> torch.Tensor:
    """Mean energy of the residual est - ref over ``w``, relative to ``l_ref``, in dB. [B]"""
    e = (w * (est - ref) ** 2).sum(-1) / w.sum(-1).clamp_min(1.0)
    return 10 * torch.log10(e / l_ref + 1e-12)


def kt1_loss(
    est: torch.Tensor,
    clean: torch.Tensor,
    act_t: torch.Tensor,
    silent: torch.Tensor,
    l_ref: torch.Tensor,
    d: torch.Tensor | None = None,
    spk_enc=None,
    weights: tuple[float, float, float] = (1.0, 0.3, 0.1),
    tau: float = 1e-3,
    floor_db: float = -30.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``act_t``: target active frames [B, F]; ``silent``: target silent frames minus the disputed
    ones [B, F]; ``l_ref`` [B]; ``d``: enrollment embeddings [B, D] for the speaker term."""
    n = est.shape[-1]
    wa, ws = frames_to_samples(act_t, n), frames_to_samples(silent, n)
    has_a, has_s = wa.sum(-1) > 0, ws.sum(-1) > 0

    snr = snr_tau(est, clean, wa, tau)
    l_act = -(snr * has_a).sum() / has_a.sum().clamp_min(1)
    sil = F.relu(residual_db(est, clean, ws, l_ref) - floor_db)
    l_sil = (sil * has_s).sum() / has_s.sum().clamp_min(1)
    l_spk = est.new_zeros(())
    if weights[2] > 0 and spk_enc is not None and bool(has_a.any()):
        e = spk_enc((est * wa)[has_a])
        l_spk = (1 - F.cosine_similarity(e, d[has_a], dim=-1)).mean()
    loss = weights[0] * l_act + weights[1] * l_sil + weights[2] * l_spk
    parts = {"snr_active": -l_act, "silence": l_sil, "spk": l_spk}
    return loss, {k: float(v.detach()) for k, v in parts.items()}
