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


class BandSep(torch.nn.Module):
    """Two-speaker band-split separator without an enrollment (the v2 band-split blocks, two
    complex masks per band): mix [B, T] -> [B, 2, T]."""

    def __init__(self, n_fft: int = 512, hop: int = 128, bands: int = 16, width: int = 128,
                 blocks: int = 6):
        super().__init__()
        from kezek.tse_v2 import BandBlock

        self.n_fft, self.hop = n_fft, hop
        self.band_sizes = [len(p) for p in torch.tensor_split(torch.arange(n_fft // 2 + 1), bands)]
        self.inp = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.LayerNorm(2 * k), torch.nn.Linear(2 * k, width))
            for k in self.band_sizes
        )
        self.blocks = torch.nn.ModuleList(BandBlock(width) for _ in range(blocks))
        self.out = torch.nn.ModuleList(torch.nn.Linear(width, 4 * k) for k in self.band_sizes)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        dtype = mix.dtype
        mix = mix.float()
        spec = torch.stft(mix, self.n_fft, self.hop, window=self.window, return_complex=True)
        feats = []
        for chunk, proj in zip(torch.split(spec, self.band_sizes, dim=1), self.inp, strict=True):
            feats.append(proj(torch.view_as_real(chunk).permute(0, 2, 1, 3).flatten(2)))
        h = torch.stack(feats, dim=1)
        for blk in self.blocks:
            h = blk(h)
        outs = []
        for feat, chunk, head in zip(h.unbind(1), torch.split(spec, self.band_sizes, dim=1),
                                     self.out, strict=True):
            m = head(feat).float().view(feat.shape[0], feat.shape[1], 2, 2, -1)  # B F src re/im k
            mask = torch.complex(m[:, :, :, 0], m[:, :, :, 1]).permute(0, 2, 3, 1)  # B src k F
            outs.append(chunk.unsqueeze(1) * mask)
        est = torch.cat(outs, dim=2)
        y = torch.istft(est.flatten(0, 1), self.n_fft, self.hop, window=self.window,
                        length=mix.shape[-1])
        return y.view(mix.shape[0], 2, -1).to(dtype)


class PretrainedSep(torch.nn.Module):
    """A public SpeechBrain separator as the block's first stage: mix [B, T] -> [B, 2, T].
    8 kHz models get a resampled input and their output resampled back to 16 kHz."""

    def __init__(self, source: str, device: str, savedir: str = "pretrained"):
        super().__init__()
        import torchaudio.functional as AF
        from speechbrain.inference.separation import SepformerSeparation

        self.AF = AF
        self.sep = SepformerSeparation.from_hparams(
            source=source, savedir=f"{savedir}/{source.split('/')[-1]}", run_opts={"device": device}
        )
        self.rate = int(self.sep.hparams.sample_rate)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        x = mix if self.rate == 16_000 else self.AF.resample(mix, 16_000, self.rate)
        y = self.sep.separate_batch(x).transpose(1, 2)
        if self.rate != 16_000:
            y = self.AF.resample(y, self.rate, 16_000)
        n = mix.shape[-1]
        return torch.nn.functional.pad(y, (0, max(0, n - y.shape[-1])))[..., :n]


class Mixture(torch.nn.Module):
    """The unprocessed input, as a reference point."""

    arch = "mixture"

    def forward(self, mix, enr, d):
        return mix
