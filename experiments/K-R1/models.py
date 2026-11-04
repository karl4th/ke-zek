"""The extractors compared in K-R1, behind one call: ``model(mix, enr, d) -> estimate``.

``mix`` [B, T] and ``enr`` [B, E] are 16 kHz waveforms; ``d`` [B, 192] is the frozen ECAPA
embedding of ``enr``. Each architecture uses what it was designed for:

* ``kezek``: KezekTSE (Conv-TasNet + FiLM on ECAPA, 4.1M; K-T0c / K-T1 checkpoints load as is);
* ``band``: the v2 band-split extractor, which encodes the enrollment waveform itself.

``net`` holds the trainable network; its state dict is what gets saved.
"""

from __future__ import annotations

import torch

from kezek.model import KezekTSE
from kezek.tse_v2 import TargetSpeakerExtractor

ARCHS = ("kezek", "band")


class Extractor(torch.nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture {arch}")
        self.arch = arch
        self.net = KezekTSE() if arch == "kezek" else TargetSpeakerExtractor()

    def forward(self, mix: torch.Tensor, enr: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.arch == "kezek":
            return self.net(mix, d)
        return self.net(mix, enr).waveform


class Mixture(torch.nn.Module):
    """The unprocessed input, as a reference point."""

    arch = "mixture"

    def forward(self, mix, enr, d):
        return mix
