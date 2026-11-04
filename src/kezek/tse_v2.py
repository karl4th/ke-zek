"""Offline target speaker extraction from a mono mixture and a voice sample."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExtractionOutput:
    waveform: torch.Tensor
    activity_logits: torch.Tensor
    mask_magnitude: torch.Tensor | None = None


class BandBlock(nn.Module):
    """Alternate temporal modeling within each band and communication across bands."""

    def __init__(self, width: int):
        super().__init__()
        self.time_norm = nn.LayerNorm(width)
        self.time_rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.time_proj = nn.Linear(width * 2, width)
        self.band_norm = nn.LayerNorm(width)
        self.band_rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.band_proj = nn.Linear(width * 2, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, bands, frames, width]
        batch, bands, frames, width = x.shape
        time_in = self.time_norm(x).reshape(batch * bands, frames, width)
        time_out, _ = self.time_rnn(time_in)
        x = x + self.time_proj(time_out).reshape(batch, bands, frames, width)

        band_in = self.band_norm(x).permute(0, 2, 1, 3).reshape(batch * frames, bands, width)
        band_out, _ = self.band_rnn(band_in)
        x = x + self.band_proj(band_out).reshape(batch, frames, bands, width).permute(0, 2, 1, 3)
        return x


class TargetSpeakerExtractor(nn.Module):
    """Band-split separator with pooled and time-varying voice-sample conditioning.

    The activity head is auxiliary. Its logits never mask the waveform output.
    Both inputs are 16 kHz mono tensors with shape [batch, samples].
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        bands: int = 8,
        width: int = 128,
        blocks: int = 4,
        enrollment_tokens: int = 32,
        conditioning: str = "mean",
    ) -> None:
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("The extractor expects 16 kHz audio")
        if n_fft < 32 or hop_length <= 0 or hop_length > n_fft:
            raise ValueError("Invalid STFT settings")
        if bands < 1 or bands > n_fft // 2 + 1:
            raise ValueError("Invalid number of bands")
        if width < 4 or width % 4 or blocks < 1 or enrollment_tokens < 1:
            raise ValueError("Invalid model size")
        if conditioning not in {"mean", "band_concat"}:
            raise ValueError("Conditioning must be mean or band_concat")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.enrollment_tokens = enrollment_tokens
        self.conditioning = conditioning
        self.band_sizes = [
            len(part) for part in torch.tensor_split(torch.arange(n_fft // 2 + 1), bands)
        ]
        self.band_inputs = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * size), nn.Linear(2 * size, width))
            for size in self.band_sizes
        )
        self.band_outputs = nn.ModuleList(nn.Linear(width, 2 * size) for size in self.band_sizes)
        self.voice_norm = nn.LayerNorm(width)
        self.voice_band_projection = (
            nn.Sequential(
                nn.LayerNorm(bands * width),
                nn.Linear(bands * width, width),
                nn.GELU(),
            )
            if conditioning == "band_concat"
            else None
        )
        self.voice_attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.voice_fusion = nn.Linear(width * 3, width)
        self.separator = nn.ModuleList(BandBlock(width) for _ in range(blocks))
        self.activity = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        # A fresh model starts as a mixture pass-through. Training must learn both
        # suppression when the target is absent and separation during overlap.
        for head in self.band_outputs:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(dtype=wav.dtype),
            return_complex=True,
        )

    def _band_features(self, spectrum: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(spectrum, self.band_sizes, dim=1)
        features = []
        for chunk, projection in zip(chunks, self.band_inputs, strict=True):
            parts = torch.view_as_real(chunk).permute(0, 2, 1, 3).flatten(2)
            features.append(projection(parts))
        return torch.stack(features, dim=1)

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> ExtractionOutput:
        if mixture.ndim != 2 or enrollment.ndim != 2:
            raise ValueError("Mixture and enrollment must have shape [batch, samples]")
        if mixture.shape[0] != enrollment.shape[0]:
            raise ValueError("Mixture and enrollment batch sizes differ")
        if mixture.shape[-1] < self.n_fft or enrollment.shape[-1] < self.n_fft:
            raise ValueError("Both waveforms must be at least one STFT window long")

        mix_spec = self._stft(mixture)
        sample_spec = self._stft(enrollment)
        mix = self._band_features(mix_spec)
        sample_bands = self._band_features(sample_spec)
        if self.voice_band_projection is None:
            sample = sample_bands.mean(dim=1)
        else:
            sample = self.voice_band_projection(
                sample_bands.permute(0, 2, 1, 3).flatten(2)
            )
        sample = self.voice_norm(sample)
        pooled_voice = sample.mean(dim=1)
        sample_tokens = F.adaptive_avg_pool1d(
            sample.transpose(1, 2), self.enrollment_tokens
        ).transpose(1, 2)

        frames = mix.mean(dim=1)
        attended_voice, _ = self.voice_attention(
            frames, sample_tokens, sample_tokens, need_weights=False
        )
        voice = self.voice_fusion(
            torch.cat([frames, attended_voice, pooled_voice[:, None, :].expand_as(frames)], dim=-1)
        )
        hidden = mix + voice[:, None]
        for block in self.separator:
            hidden = block(hidden)

        activity_logits = self.activity(hidden.mean(dim=1)).squeeze(-1)
        estimates = []
        magnitudes = []
        for features, spectrum, head in zip(
            hidden.unbind(dim=1),
            torch.split(mix_spec, self.band_sizes, dim=1),
            self.band_outputs,
            strict=True,
        ):
            delta = head(features)
            real, imag = delta.chunk(2, dim=-1)
            magnitude = 2 * torch.sigmoid(real)
            mask = torch.complex(magnitude, torch.tanh(imag))
            estimates.append(spectrum * mask.transpose(1, 2))
            magnitudes.append(magnitude.transpose(1, 2))
        target_spec = torch.cat(estimates, dim=1)
        target = torch.istft(
            target_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(dtype=mixture.dtype),
            length=mixture.shape[-1],
        )
        return ExtractionOutput(
            waveform=target,
            activity_logits=activity_logits,
            mask_magnitude=torch.cat(magnitudes, dim=1),
        )
