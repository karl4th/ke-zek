"""Frame-level target routing over the two outputs of a frozen separator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(features)
        residual = self.pointwise(F.gelu(residual))
        return self.norm(features + residual)


class TargetRouter(nn.Module):
    """Select either separated stream or silence at each short audio frame."""

    def __init__(self, embedding_dim: int = 192, channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(256), persistent=False)
        self.input = nn.Conv1d(258, channels, 1)
        self.condition = nn.Linear(embedding_dim * 3 + 2, channels * 2)
        self.blocks = nn.Sequential(*(
            _TemporalBlock(channels, dilation)
            for dilation in (1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)
        ))
        self.head = nn.Conv1d(channels, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        sources: torch.Tensor,
        enrollment: torch.Tensor,
        source_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sources.ndim != 3 or sources.shape[-1] != 2:
            raise ValueError("Sources must have shape [batch, samples, 2]")
        if source_embeddings.shape != (len(sources), 2, enrollment.shape[-1]):
            raise ValueError("Source embeddings must match enrollment and batch")
        batch, samples, _ = sources.shape
        normalized_enrollment = F.normalize(enrollment, dim=-1)
        normalized_sources = F.normalize(source_embeddings, dim=-1)
        cosines = (normalized_enrollment[:, None] * normalized_sources).sum(-1)
        condition = torch.cat((
            normalized_enrollment,
            normalized_sources.flatten(1), cosines,
        ), dim=-1)
        flat = sources.transpose(1, 2).reshape(batch * 2, samples)
        magnitude = torch.stft(
            flat, n_fft=256, hop_length=64, window=self.window,
            return_complex=True,
        ).abs()
        features = torch.log1p(magnitude).reshape(batch, 258, -1)
        features = self.input(features)
        gain, bias = self.condition(condition).chunk(2, dim=-1)
        features = features * (1 + gain[..., None]) + bias[..., None]
        logits = self.head(self.blocks(features))
        logits = logits + torch.cat((
            8 * cosines, torch.full_like(cosines[:, :1], -2),
        ), dim=-1)[..., None]
        probabilities = F.interpolate(
            logits.softmax(dim=1), size=samples, mode="linear", align_corners=False,
        )
        estimate = (
            probabilities[:, :2].transpose(1, 2) * sources
        ).sum(dim=-1)
        return estimate, logits


def oracle_route_labels(
    sources: torch.Tensor, target: torch.Tensor, frames: int,
) -> torch.Tensor:
    """Supervise the best source or silence for each frame from clean references."""
    if sources.shape[:2] != target.shape or sources.shape[-1] != 2:
        raise ValueError("Target and sources must have matching audio lengths")
    candidates = torch.cat((sources, torch.zeros_like(target[..., None])), dim=-1)
    errors = (candidates - target[..., None]).square().transpose(1, 2)
    frame_errors = F.adaptive_avg_pool1d(errors, frames)
    return frame_errors.argmin(dim=1)
