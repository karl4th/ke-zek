"""Two-block Kezek, M2S v0.1 layout.

Block 1, ``IdentityEncoder``: 30 s mono context -> frame-level activity of two speaker slots
(independent sigmoids: both may be active) and two unit-normalized slot embeddings. With
``speaker`` (a frozen pretrained speaker encoder, ECAPA on VoxCeleb), each slot embedding is
that encoder applied to the mixture weighted by the slot's solo activity: Open Yap has ~150
voices, too few to learn speaker identity from scratch (E2: same-span identity accuracy 0.65 for
the learned 128-D head). Without it, the learned head pools frame embeddings over solo frames.
``embed_single`` embeds single-speaker audio (clean references, separator outputs).

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


class FrozenFeatures(nn.Module):
    """Frozen pretrained speech features on the 20 ms grid: [B, T] at 16 kHz -> [B, D, F20].

    * ``wavlm``: WavLM Base+ (torchaudio), learned softmax-weighted sum of its layers, 50 Hz;
    * ``mimi``: Mimi codec encoder latents before quantization (12.5 Hz, 512-D), input
      resampled to 24 kHz, each frame repeated 4x onto the 20 ms grid.
    The backbone is never trained and not saved with the encoder (only the layer weights are).
    """

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "wavlm":
            import torchaudio

            bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
            self._net = [bundle.get_model().eval()]
            self.dim = 768
            self.layer_w = nn.Parameter(torch.zeros(13))
        elif kind == "mimi":
            from transformers import MimiModel

            self._net = [MimiModel.from_pretrained("kyutai/mimi").eval()]
            self.dim = 512
        else:
            raise ValueError(kind)
        for p in self._net[0].parameters():
            p.requires_grad_(False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        net = self._net[0]
        if next(net.parameters()).device != wav.device:  # kept outside the module tree
            net.to(wav.device)
        n20 = wav.shape[-1] // (2 * HOP)
        with torch.no_grad(), torch.autocast(wav.device.type, enabled=False):
            if self.kind == "wavlm":
                layers, _ = net.extract_features(wav.float())
                hs = torch.stack(layers, 0)  # [L, B, F, D]
            else:
                import torchaudio.functional as AF

                x = AF.resample(wav.float(), 16_000, 24_000)[:, None]
                e = net.encoder(x)
                e = net.encoder_transformer(e.transpose(1, 2)).last_hidden_state.transpose(1, 2)
                e = net.downsample(e)  # [B, 512, F12.5]
        if self.kind == "wavlm":
            f = (torch.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0).transpose(1, 2)
        else:
            f = e.repeat_interleave(4, dim=-1)
        if f.shape[-1] < n20:
            f = F.pad(f, (0, n20 - f.shape[-1]), mode="replicate")
        return f[..., :n20].float()


class IdentityEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, width: int = 256, layers: int = 3, emb: int = 128, speaker=None,
                 features: str = "mel"):
        super().__init__()
        self._speaker = [speaker]  # not a submodule: frozen, not saved with the encoder
        self.solo_threshold = 0.0  # > 0: the speaker encoder sees only confidently solo frames
        self.dim = speaker.dim if speaker is not None else emb
        self.features = features
        if features == "mel":
            import torchaudio

            self.mel = torchaudio.transforms.MelSpectrogram(16_000, n_fft=400, hop_length=HOP, n_mels=n_mels)
            self.front = nn.Sequential(
                nn.Conv1d(n_mels, width, 5, padding=2), nn.GELU(),
                nn.Conv1d(width, width, 3, stride=2, padding=1), nn.GELU(),
            )
        else:
            self.ssl = FrozenFeatures(features)
            self.front = nn.Sequential(
                nn.Conv1d(self.ssl.dim, width, 3, padding=1), nn.GELU(),
            )
        self.rnn = nn.LSTM(width, width, num_layers=layers, batch_first=True, bidirectional=True)
        self.act = nn.Linear(2 * width, 2)
        self.emb = nn.Linear(2 * width, emb)

    def frames(self, wav: torch.Tensor):
        if self.features == "mel":
            with torch.autocast(wav.device.type, enabled=False):
                m = self.mel(wav.float()).clamp_min(1e-8).log()
                m = m - m.mean(dim=-1, keepdim=True)
        else:
            m = self.ssl(wav)
            m = (m - m.mean(-1, keepdim=True)) / (m.std(-1, keepdim=True) + 1e-5)
        h = self.front(m).transpose(1, 2)
        with torch.autocast(wav.device.type, enabled=False):  # no fast bf16 LSTM kernel
            h, _ = self.rnn(h.float())
        return self.act(h).transpose(1, 2).float(), self.emb(h).float()

    def forward(self, wav: torch.Tensor):
        """wav [B, T] -> activity logits [B, 2, F], slot embeddings [B, 2, dim] (unit norm)."""
        logits, e = self.frames(wav)
        p = torch.sigmoid(logits)
        solo = torch.stack([p[:, 0] * (1 - p[:, 1]), p[:, 1] * (1 - p[:, 0])], dim=1)
        spk = self._speaker[0]
        if spk is not None:
            T = wav.shape[-1]
            ws = (solo > self.solo_threshold).float() if self.solo_threshold > 0 else solo
            w = ws.repeat_interleave(2 * HOP, dim=-1)[..., :T]
            w = F.pad(w, (0, T - w.shape[-1]))
            masked = (wav[:, None] * w).flatten(0, 1)
            with torch.autocast(wav.device.type, enabled=False):
                z = spk(masked.float()).view(wav.shape[0], 2, -1)
            return logits, F.normalize(z, dim=-1)
        w = solo / (solo.sum(-1, keepdim=True) + 1e-4)
        z = torch.einsum("bsf,bfd->bsd", w, e)
        return logits, F.normalize(z, dim=-1)

    def embed_single(self, wav: torch.Tensor) -> torch.Tensor:
        """Embedding of single-speaker audio [B, T] -> [B, 128]: frames weighted by the larger
        slot activity (the one voice present)."""
        spk = self._speaker[0]
        if spk is not None:
            with torch.autocast(wav.device.type, enabled=False):
                return F.normalize(spk(wav.float()), dim=-1)
        logits, e = self.frames(wav)
        w = torch.sigmoid(logits).amax(1)
        w = w / (w.sum(-1, keepdim=True) + 1e-4)
        return F.normalize(torch.einsum("bf,bfd->bd", w, e), dim=-1)


class ConditionedSeparator(nn.Module):
    def __init__(self, emb: int = 128):
        super().__init__()
        self.tse = KezekTSE()
        cond = self.tse.d_proj[0].in_features
        self.proj = nn.Linear(emb, cond)
        if emb == cond:  # slots already in the separator's (ECAPA) space: start as identity
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(cond))
                self.proj.bias.zero_()

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
