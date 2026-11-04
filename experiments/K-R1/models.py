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


class SepTasNet(torch.nn.Module):
    """Two-speaker separator without an enrollment: the KezekTSE backbone with its speaker input
    held at zero and a mask per output. Loading a KezekTSE state dict into ``net`` reuses its
    encoder, TCN and decoder (the block's first stage)."""

    def __init__(self):
        super().__init__()
        self.net = KezekTSE()
        nf, bn = self.net.enc.out_channels, self.net.bottleneck.out_channels
        self.masks = torch.nn.Sequential(
            torch.nn.PReLU(), torch.nn.Conv1d(bn, 2 * nf, 1), torch.nn.Sigmoid()
        )

    def init_masks_from_net(self):
        """Both output masks start as the loaded single-target mask."""
        conv = self.net.mask[1]
        with torch.no_grad():
            self.masks[1].weight.copy_(torch.cat([conv.weight, conv.weight]))
            self.masks[1].bias.copy_(torch.cat([conv.bias, -conv.bias]))

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """mix [B, T] -> sources [B, 2, T]."""
        net = self.net
        T = mix.shape[-1]
        pad = (-(T - net.kernel) % net.stride) if T >= net.kernel else net.kernel - T
        x = torch.nn.functional.pad(mix, (0, pad)).unsqueeze(1)
        w = torch.relu(net.enc(x))
        h = net.bottleneck(net.norm(w))
        d = net.d_proj(mix.new_zeros(mix.shape[0], net.d_proj[0].in_features))
        for blk in net.blocks:
            h = blk(h, d)
        m = self.masks(h).view(mix.shape[0], 2, w.shape[1], w.shape[2])
        y = net.dec((w.unsqueeze(1) * m).flatten(0, 1)).squeeze(1)
        return y.view(mix.shape[0], 2, -1)[..., :T]


class Mixture(torch.nn.Module):
    """The unprocessed input, as a reference point."""

    arch = "mixture"

    def forward(self, mix, enr, d):
        return mix
