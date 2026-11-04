"""Enrollment-conditioned two-source separation with a pretrained backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedSeparator(nn.Module):
    """Modulate SepFormer encoder features so output zero follows the target voice."""

    def __init__(self, encoder, masknet, decoder, channels: int = 256, voice_dim: int = 192):
        super().__init__()
        self.encoder = encoder
        self.masknet = masknet
        self.decoder = decoder
        self.condition = nn.Linear(voice_dim, channels * 2)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, mixture: torch.Tensor, voice: torch.Tensor) -> torch.Tensor:
        if mixture.ndim != 2 or voice.shape != (len(mixture), self.condition.in_features):
            raise ValueError("Mixture and voice embedding batch shapes do not match")
        encoded = self.encoder(mixture)
        gain, bias = self.condition(F.normalize(voice, dim=-1)).chunk(2, dim=-1)
        conditioned = encoded * (1 + gain[..., None]) + bias[..., None]
        masks = self.masknet(conditioned)
        if masks.shape[:2] != (2, len(mixture)):
            raise ValueError("Mask network must return two source masks")
        estimates = torch.stack([
            self.decoder(encoded * masks[side]) for side in range(2)
        ], dim=-1)
        if estimates.shape[1] < mixture.shape[1]:
            estimates = F.pad(estimates, (0, 0, 0, mixture.shape[1] - estimates.shape[1]))
        return estimates[:, : mixture.shape[1]]
