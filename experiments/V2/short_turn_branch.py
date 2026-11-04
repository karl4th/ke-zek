"""A gated, multi-scale temporal correction to a target extraction mask."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalMaskCorrection(nn.Module):
    """Preserve a fixed mask head and learn local corrections around speech events."""

    def __init__(self, base: nn.Linear, width: int = 128) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("The existing mask projection must be linear")
        self.base = base.requires_grad_(False)
        channels = base.in_features
        self.norm = nn.LayerNorm(channels)
        self.short = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        self.long = nn.Conv1d(channels, channels, 11, padding=5, groups=channels)
        self.reduce = nn.Conv1d(channels * 2, width, 1)
        self.delta = nn.Conv1d(width, base.out_features, 1)
        self.gate = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 3 or frames.shape[-1] != self.base.in_features:
            raise ValueError("Expected [batch, time, hidden] mask features")
        x = self.norm(frames).transpose(1, 2)
        events = F.gelu(self.reduce(torch.cat((self.short(x), self.long(x)), dim=1)))
        correction = torch.sigmoid(self.gate(events)) * self.delta(events)
        return self.base(frames) + correction.transpose(1, 2)


def attach_branch(head: nn.Module, width: int = 128) -> TemporalMaskCorrection:
    """Replace the single mask projection while retaining its frozen weights."""
    linear = head.model.linear[0]
    branch = TemporalMaskCorrection(linear, width=width).to(next(linear.parameters()).device)
    head.model.linear[0] = branch
    return branch
