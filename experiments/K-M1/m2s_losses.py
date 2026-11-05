"""M2S v0.1 objectives with explicit reductions.

Every raw term is returned per sequence ([B]) or per sequence and permutation ([B, 2]); the
trainer applies the calibrated weights, chooses one permutation per sequence, and logs raw and
weighted values separately.

* L_sep  = -SI-SDR(y, s) on the active part of the target, fp32 with eps, in units of 10 dB,
           plus multi-resolution STFT magnitude L1 (scale-sensitive; normalized by the mixture's
           magnitude, since the target's is ~0 when that person is silent). Mean over valid windows
           (target active >= 0.5 s) for SI-SDR, over all windows for the STFT term.
* L_act  = BCE of slot activity against source activity (20 ms frames) plus the bounded
           energy penalty of an output where its speaker is silent:
           clamp(10 log10(E_out / L_ref) + 30, 0, 30) / 30, edges excluded.
* L_id   = InfoNCE between mono slot embeddings and clean-reference embeddings of the same
           speakers taken from another window (other participants in the batch are negatives;
           temperature 0.1), plus 1 - cos(embedding of output, its slot) on chunks where the
           speaker talks alone >= 1 s.
* L_mix  = |x - (y_A + y_B)|_1 / |x|_1 over samples where anyone speaks (perm-invariant).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FRAME = 160
RESOLUTIONS = ((512, 128), (1024, 256), (256, 64))


def to_samples(mask: torch.Tensor, n: int) -> torch.Tensor:
    w = mask.float().repeat_interleave(FRAME, dim=-1)
    if w.shape[-1] < n:
        w = F.pad(w, (0, n - w.shape[-1]))
    return w[..., :n]


def neg_sisdr_active(y, s, act, eps=1e-8):
    """[N, n] waveforms, [N, F] activity -> (-SI-SDR / 10 dB [N], valid [N])."""
    w = to_samples(act, y.shape[-1])
    yy, ss = (y * w).float(), (s * w).float()
    alpha = (yy * ss).sum(-1, keepdim=True) / ((ss * ss).sum(-1, keepdim=True) + eps)
    proj = alpha * ss
    si = 10 * torch.log10((proj * proj).sum(-1) / (((yy - proj) ** 2).sum(-1) + eps) + eps)
    valid = act.sum(-1) >= 50
    return -si.clamp(-50, 50) / 10, valid  # clamp only guards Inf/NaN on degenerate windows


def mr_stft_l1(y, s, x, eps=1e-6):
    """Mean |Y - S| over time-frequency, per window, normalized by the mean |X| of the mixture
    (the target's own magnitude is ~0 when that person is silent)."""
    out = 0.0
    for n_fft, hop in RESOLUTIONS:
        win = torch.hann_window(n_fft, device=y.device)
        Y, S, X = (torch.stft(v.float(), n_fft, hop, window=win, return_complex=True).abs() for v in (y, s, x))
        out = out + (Y - S).abs().mean((-1, -2)) / (X.mean((-1, -2)) + eps)
    return out / len(RESOLUTIONS)


def silence_energy(y, act, disputed, l_ref):
    """Bounded penalty of output energy on frames where its speaker is silent. [N], valid [N]."""
    sil = (~act.bool()) & (~disputed.bool())
    w = to_samples(sil, y.shape[-1])
    e = (w * y.float() ** 2).sum(-1) / w.sum(-1).clamp_min(1.0)
    pen = (10 * torch.log10(e / l_ref + 1e-10) + 30).clamp(0, 30) / 30
    return pen, sil.sum(-1) >= 20


def masked_mean(x, valid):
    v = valid.float()
    return (x * v).sum(-1) / v.sum(-1).clamp_min(1.0)


def mixture_l1(x, ysum, act_any):
    w = to_samples(act_any, x.shape[-1])
    return (w * (x - ysum).abs()).sum(-1) / ((w * x.abs()).sum(-1) + 1e-6)


def info_nce(z, r, tau=0.1):
    """z [B, 2, D] mono slot embeddings, r [B, 2, D] clean references from another window.
    Returns [B, 2]: loss for slot i <- speaker i (perm 0) and slot i <- speaker 1-i (perm 1)."""
    B = z.shape[0]
    keys = r.reshape(2 * B, -1)
    logits = z.reshape(2 * B, -1) @ keys.T / tau  # [2B, 2B]
    logp = logits.log_softmax(-1).view(B, 2, B, 2)
    idx = torch.arange(B, device=z.device)
    same = -(logp[idx, 0, idx, 0] + logp[idx, 1, idx, 1]) / 2
    cross = -(logp[idx, 0, idx, 1] + logp[idx, 1, idx, 0]) / 2
    return torch.stack([same, cross], dim=1)
