"""Scale-aware training loss for direct target speaker extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kezek.losses import snr_tau
from kezek.tse_v2 import ExtractionOutput

FRAME = 160


def _activity(waveform: torch.Tensor) -> torch.Tensor:
    frames = waveform.unfold(-1, FRAME, FRAME)
    rms = frames.square().mean(-1).sqrt()
    peak = rms.amax(dim=-1, keepdim=True)
    return rms > torch.maximum(peak * 0.01, torch.full_like(peak, 1e-4))


def _sample_mask(frames: torch.Tensor, length: int) -> torch.Tensor:
    return frames.repeat_interleave(FRAME, dim=-1).to(torch.float32)[..., :length]


def extraction_loss(
    output: ExtractionOutput, target: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reward target fidelity, overlap recovery, and suppression when target is absent."""
    estimate = output.waveform
    if estimate.shape != target.shape or target.shape != other.shape:
        raise ValueError("Estimate, target, and other must have matching shapes")
    if target.shape[-1] % FRAME:
        raise ValueError("Training windows must be a multiple of 10 ms at 16 kHz")

    active, interfering = _activity(target), _activity(other)
    overlap = active & interfering
    protected = F.max_pool1d(active[:, None].float(), kernel_size=11, stride=1, padding=5)[
        :, 0
    ].bool()
    absent = ~protected & interfering
    n = target.shape[-1]
    active_mask = _sample_mask(active, n)
    overlap_mask = _sample_mask(overlap, n)
    absent_mask = _sample_mask(absent, n)

    active_snr = snr_tau(estimate, target, active_mask)
    has_active = active.any(-1)
    active_term = -(active_snr * has_active).sum() / has_active.sum().clamp_min(1)
    overlap_snr = snr_tau(estimate, target, overlap_mask)
    has_overlap = overlap.any(-1)
    overlap_term = -(overlap_snr * has_overlap).sum() / has_overlap.sum().clamp_min(1)

    residual_power = (absent_mask * (estimate - target).square()).sum(-1)
    reference_power = (absent_mask * other.square()).sum(-1)
    leakage_db = 10 * torch.log10((residual_power + 1e-8) / (reference_power + 1e-8))
    has_absent = absent.any(-1)
    silence_term = (F.relu(leakage_db + 30.0) * has_absent).sum() / has_absent.sum().clamp_min(1)

    activity_labels = F.interpolate(
        active[:, None].float(), size=output.activity_logits.shape[-1], mode="nearest"
    )[:, 0]
    activity_term = F.binary_cross_entropy_with_logits(output.activity_logits, activity_labels)
    loss = active_term + 0.5 * overlap_term + 0.2 * silence_term + 0.05 * activity_term
    parts = {
        "active_snr_db": -float(active_term.detach()),
        "overlap_snr_db": -float(overlap_term.detach()),
        "absent_residual_db": float(
            ((leakage_db * has_absent).sum() / has_absent.sum().clamp_min(1)).detach()
        ),
        "activity_bce": float(activity_term.detach()),
    }
    return loss, parts
