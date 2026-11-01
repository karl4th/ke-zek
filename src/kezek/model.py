"""Kezek target speaker extractor, audio variant (docs/architecture.md).

    enrollment ─► SpeakerEncoder (frozen ECAPA) ─► d ─────────────┐ FiLM in every block
    mixture ─► Conv encoder ─► [TCN block × N, conditioned on d] ─► mask ─► decoder ─► ŝ

Loss: ``-SI-SDR(ŝ, s) + spk_weight * (1 - cos(SE(ŝ), d))``. The speaker term keeps the model
on the voice of the sample instead of the louder speaker. It needs the encoder to be frozen
and the same one that produced ``d``.

torch is imported here only; the rest of the package (mixing, metrics) needs just numpy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    """Frozen pretrained ECAPA-TDNN (SpeechBrain, VoxCeleb). 16 kHz mono in, ``dim`` out."""

    dim = 192

    def __init__(
        self, source: str = "speechbrain/spkrec-ecapa-voxceleb", savedir: str | None = None, device: str = "cuda"
    ):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier

        self.enc = EncoderClassifier.from_hparams(
            source=source, savedir=savedir or "pretrained/ecapa", run_opts={"device": device}
        )
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()

    def train(self, mode: bool = True):  # always eval: frozen statistics
        super().train(False)
        return self

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] -> embedding [B, dim], differentiable w.r.t. ``wav``."""
        emb = self.enc.encode_batch(wav, normalize=False)  # [B, 1, dim]
        return emb.squeeze(1)


class GlobalLN(nn.Module):
    """Global layer norm over (channels, time)."""

    def __init__(self, ch: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, ch, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(1, 2), keepdim=True)
        v = ((x - m) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.g * (x - m) / torch.sqrt(v + 1e-8) + self.b


class FiLMBlock(nn.Module):
    """TCN block: 1x1 -> PReLU/norm -> FiLM(d) -> dilated depthwise -> PReLU/norm -> 1x1 + residual."""

    def __init__(self, bn: int, hid: int, kernel: int, dilation: int, cond: int):
        super().__init__()
        self.inp = nn.Conv1d(bn, hid, 1)
        self.act1, self.norm1 = nn.PReLU(), GlobalLN(hid)
        self.film = nn.Linear(cond, 2 * hid)
        pad = (kernel - 1) * dilation // 2
        self.dw = nn.Conv1d(hid, hid, kernel, padding=pad, dilation=dilation, groups=hid)
        self.act2, self.norm2 = nn.PReLU(), GlobalLN(hid)
        self.out = nn.Conv1d(hid, bn, 1)
        nn.init.zeros_(self.film.weight)  # start as identity: gamma=1, beta=0
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.act1(self.inp(x)))
        gamma, beta = self.film(d).unsqueeze(-1).chunk(2, dim=1)
        h = (1 + gamma) * h + beta
        h = self.norm2(self.act2(self.dw(h)))
        return x + self.out(h)


class KezekTSE(nn.Module):
    """Conv-TasNet backbone with speaker conditioning in every block (~4M parameters)."""

    def __init__(
        self,
        n_filters: int = 256,
        kernel: int = 32,
        bottleneck: int = 128,
        hidden: int = 256,
        blocks: int = 8,
        repeats: int = 3,
        tcn_kernel: int = 3,
        cond_dim: int = SpeakerEncoder.dim,
    ):
        super().__init__()
        self.kernel, self.stride = kernel, kernel // 2
        self.enc = nn.Conv1d(1, n_filters, kernel, stride=self.stride, bias=False)
        self.norm = GlobalLN(n_filters)
        self.bottleneck = nn.Conv1d(n_filters, bottleneck, 1)
        self.d_proj = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.PReLU())
        self.blocks = nn.ModuleList(
            FiLMBlock(bottleneck, hidden, tcn_kernel, 2**i, cond_dim)
            for _ in range(repeats)
            for i in range(blocks)
        )
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(bottleneck, n_filters, 1), nn.Sigmoid())
        self.dec = nn.ConvTranspose1d(n_filters, 1, kernel, stride=self.stride, bias=False)

    def forward(self, mix: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """mix [B, T], speaker embedding d [B, cond_dim] -> estimate [B, T]."""
        T = mix.shape[-1]
        pad = (-(T - self.kernel) % self.stride) if T >= self.kernel else self.kernel - T
        x = F.pad(mix, (0, pad)).unsqueeze(1)
        w = F.relu(self.enc(x))
        h = self.bottleneck(self.norm(w))
        d = self.d_proj(F.normalize(d, dim=-1) * d.shape[-1] ** 0.5)
        for blk in self.blocks:
            h = blk(h, d)
        est = self.dec(w * self.mask(h)).squeeze(1)
        return est[..., :T]


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative SI-SDR in dB, per item [B]."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / ((ref**2).sum(-1, keepdim=True) + eps)
    noise = est - proj
    return -10 * torch.log10((proj**2).sum(-1) / ((noise**2).sum(-1) + eps) + eps)


def kezek_loss(
    est: torch.Tensor, ref: torch.Tensor, d: torch.Tensor, spk_enc: SpeakerEncoder, spk_weight: float = 0.1
) -> tuple[torch.Tensor, dict[str, float]]:
    sisdr = si_sdr_loss(est, ref).mean()
    if spk_weight > 0:
        spk = (1 - F.cosine_similarity(spk_enc(est), d, dim=-1)).mean()
    else:
        spk = est.new_zeros(())
    return sisdr + spk_weight * spk, {"neg_sisdr": float(sisdr), "spk": float(spk)}
