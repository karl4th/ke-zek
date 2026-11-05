"""Two-block Kezek, M2S v0.1 layout.

Block 1, ``IdentityEncoder``: 30 s mono context -> frame-level activity of two speaker slots
(independent sigmoids: both may be active) and two unit-normalized 128-D slot embeddings, each
the activity-weighted mean of frame embeddings over frames where that slot talks alone.
``embed_single`` gives the embedding of single-speaker audio (clean references, separator
outputs) with the same network.

Block 2, ``ConditionedSeparator``: noncausal Conv-TasNet TCN with FiLM conditioning in every
block (the KezekTSE network; K-T0c weights load into ``tse``); the 128-D slot embedding is
projected to the FiLM input. One shared conditional extractor, run once per slot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kezek.model import KezekTSE

HOP = 160  # 10 ms mel hop; frames are 20 ms after the strided convolution


class IdentityEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, width: int = 256, layers: int = 3, emb: int = 128):
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(16_000, n_fft=400, hop_length=HOP, n_mels=n_mels)
        self.front = nn.Sequential(
            nn.Conv1d(n_mels, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1), nn.GELU(),
        )
        self.rnn = nn.LSTM(width, width, num_layers=layers, batch_first=True, bidirectional=True)
        self.act = nn.Linear(2 * width, 2)
        self.emb = nn.Linear(2 * width, emb)

    def frames(self, wav: torch.Tensor):
        with torch.autocast(wav.device.type, enabled=False):
            m = self.mel(wav.float()).clamp_min(1e-8).log()
            m = m - m.mean(dim=-1, keepdim=True)
        h = self.front(m).transpose(1, 2)
        with torch.autocast(wav.device.type, enabled=False):  # no fast bf16 LSTM kernel
            h, _ = self.rnn(h.float())
        return self.act(h).transpose(1, 2).float(), self.emb(h).float()

    def forward(self, wav: torch.Tensor):
        """wav [B, T] -> activity logits [B, 2, F], slot embeddings [B, 2, 128] (unit norm)."""
        logits, e = self.frames(wav)
        p = torch.sigmoid(logits)
        solo = torch.stack([p[:, 0] * (1 - p[:, 1]), p[:, 1] * (1 - p[:, 0])], dim=1)
        w = solo / (solo.sum(-1, keepdim=True) + 1e-4)
        z = torch.einsum("bsf,bfd->bsd", w, e)
        return logits, F.normalize(z, dim=-1)

    def embed_single(self, wav: torch.Tensor) -> torch.Tensor:
        """Embedding of single-speaker audio [B, T] -> [B, 128]: frames weighted by the larger
        slot activity (the one voice present)."""
        logits, e = self.frames(wav)
        w = torch.sigmoid(logits).amax(1)
        w = w / (w.sum(-1, keepdim=True) + 1e-4)
        return F.normalize(torch.einsum("bf,bfd->bd", w, e), dim=-1)


class ConditionedSeparator(nn.Module):
    def __init__(self, emb: int = 128):
        super().__init__()
        self.tse = KezekTSE()
        self.proj = nn.Linear(emb, self.tse.d_proj[0].in_features)

    def forward(self, mix: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.tse(mix, self.proj(z))


def frames_20ms(act_10ms: torch.Tensor, n: int) -> torch.Tensor:
    a = act_10ms.float()
    if a.shape[-1] % 2:
        a = F.pad(a, (0, 1))
    a = a.unflatten(-1, (-1, 2)).amax(-1)
    if a.shape[-1] < n:
        a = F.pad(a, (0, n - a.shape[-1]))
    return a[..., :n]
