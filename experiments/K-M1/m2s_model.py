"""Two-block Kezek (M2S layout).

Block 1, ``IdentityEncoder``: log-mel of a mono span (30 s context) -> per-frame activity of two
speaker slots (independent sigmoids, overlap allowed) and two slot embeddings, each a weighted
mean of frame embeddings over frames where that slot talks alone. Embeddings live in the ECAPA
space (192-d) so that block 2 can start from a KezekTSE trained on ECAPA conditioning.

Block 2, ``KezekTSE``: the conditioned separator, run once per slot on 4 s chunks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_HOP = 160  # 10 ms mel hop; the encoder works at 20 ms after subsampling


class IdentityEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, width: int = 256, layers: int = 3, emb: int = 192):
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            16_000, n_fft=400, hop_length=FRAME_HOP, n_mels=n_mels
        )
        self.front = nn.Sequential(
            nn.Conv1d(n_mels, width, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1),  # 20 ms frames
            nn.GELU(),
        )
        self.rnn = nn.LSTM(width, width, num_layers=layers, batch_first=True, bidirectional=True)
        self.act = nn.Linear(2 * width, 2)
        self.emb = nn.Linear(2 * width, emb)

    def forward(self, wav: torch.Tensor):
        """wav [B, T] -> activity logits [B, 2, F] at 20 ms, slot embeddings [B, 2, emb]."""
        with torch.autocast(wav.device.type, enabled=False):
            m = self.mel(wav.float()).clamp_min(1e-8).log()
            m = m - m.mean(dim=-1, keepdim=True)
        h = self.front(m).transpose(1, 2)
        h, _ = self.rnn(h)
        logits = self.act(h).transpose(1, 2)
        e = self.emb(h)
        p = torch.sigmoid(logits.float())
        solo = torch.stack([p[:, 0] * (1 - p[:, 1]), p[:, 1] * (1 - p[:, 0])], dim=1)
        w = solo / (solo.sum(-1, keepdim=True) + 1e-4)
        z = torch.einsum("bsf,bfd->bsd", w.to(e.dtype), e)
        return logits, z


def frames_20ms(act_10ms: torch.Tensor, n: int) -> torch.Tensor:
    """10 ms activity labels [..., F10] -> the encoder's 20 ms grid of length n (any frame
    within a 20 ms step active = active)."""
    a = act_10ms.float()
    if a.shape[-1] % 2:
        a = F.pad(a, (0, 1))
    a = a.unflatten(-1, (-1, 2)).amax(-1)
    if a.shape[-1] < n:
        a = F.pad(a, (0, n - a.shape[-1]))
    return a[..., :n]
