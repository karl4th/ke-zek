"""Spectral U-Net target speaker extractor with an explicit enrollment profile."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kezek.tse_v2 import ExtractionOutput


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.layers(audio)


class SpectralUNetExtractor(nn.Module):
    """Estimate a complex mask from a mixture and the target's clean spectral profile."""

    def __init__(
        self, sample_rate: int = 16_000, n_fft: int = 512, hop_length: int = 128
    ) -> None:
        super().__init__()
        if sample_rate != 16_000 or n_fft < 32 or hop_length < 1 or hop_length > n_fft:
            raise ValueError("Invalid spectral settings")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        frequencies = n_fft // 2 + 1
        self.profile = nn.Sequential(
            nn.LayerNorm(frequencies),
            nn.Linear(frequencies, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.down1 = ConvBlock(4, 32)
        self.down2 = ConvBlock(32, 64, stride=2)
        self.down3 = ConvBlock(64, 128, stride=2)
        self.middle = ConvBlock(128, 128)
        self.film2 = nn.Linear(128, 128)
        self.film3 = nn.Linear(128, 256)
        self.up2 = ConvBlock(192, 64)
        self.up1 = ConvBlock(96, 32)
        self.mask = nn.Conv2d(32, 2, 1)
        self.activity = nn.Conv1d(32, 1, 1)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)

    def encode_voice(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim != 2 or audio.shape[-1] < self.n_fft:
            raise ValueError("Voice audio must have shape [batch, samples] and one STFT window")
        spectrum = torch.stft(
            audio,
            self.n_fft,
            self.hop_length,
            window=self.window.to(dtype=audio.dtype),
            return_complex=True,
        )
        profile = torch.log1p(spectrum.abs()).mean(dim=-1)
        return self.profile(profile), profile

    @staticmethod
    def _condition(features: torch.Tensor, projection: nn.Linear, voice: torch.Tensor):
        gain, bias = projection(voice).chunk(2, dim=-1)
        return features * (1 + gain.tanh()[:, :, None, None]) + bias[:, :, None, None]

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> ExtractionOutput:
        if mixture.ndim != 2 or enrollment.ndim != 2:
            raise ValueError("Mixture and enrollment must have shape [batch, samples]")
        if mixture.shape[0] != enrollment.shape[0]:
            raise ValueError("Mixture and enrollment batch sizes differ")
        if min(mixture.shape[-1], enrollment.shape[-1]) < self.n_fft:
            raise ValueError("Both waveforms must be at least one STFT window long")
        window = self.window.to(dtype=mixture.dtype)
        spectrum = torch.stft(
            mixture, self.n_fft, self.hop_length, window=window, return_complex=True
        )
        voice, voice_profile = self.encode_voice(enrollment)
        features = torch.stack(
            (
                spectrum.real,
                spectrum.imag,
                torch.log1p(spectrum.abs()),
                voice_profile[:, :, None].expand_as(spectrum.real),
            ),
            dim=1,
        )
        first = self.down1(features)
        second = self._condition(self.down2(first), self.film2, voice)
        third = self._condition(self.down3(second), self.film3, voice)
        hidden = self.middle(third)
        hidden = self.up2(torch.cat((F.interpolate(hidden, size=second.shape[-2:]), second), 1))
        hidden = self.up1(torch.cat((F.interpolate(hidden, size=first.shape[-2:]), first), 1))
        real, imag = self.mask(hidden).unbind(dim=1)
        complex_mask = torch.complex(2 * real.sigmoid(), imag.tanh())
        estimate = torch.istft(
            spectrum * complex_mask,
            self.n_fft,
            self.hop_length,
            window=window,
            length=mixture.shape[-1],
        )
        activity_logits = self.activity(hidden.mean(dim=2)).squeeze(1)
        return ExtractionOutput(waveform=estimate, activity_logits=activity_logits)
