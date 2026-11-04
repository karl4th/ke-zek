"""Chunked, sample-aligned inference for a trained direct extractor."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from kezek.tse_v2 import TargetSpeakerExtractor
from kezek.tse_v2_unet import SpectralUNetExtractor

SAMPLE_RATE = 16_000


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        audio, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


def extract_chunks(
    model: TargetSpeakerExtractor | SpectralUNetExtractor,
    mixture: np.ndarray,
    enrollment: np.ndarray,
    device: str,
    chunk_samples: int = 64_000,
    overlap_samples: int = 16_000,
) -> np.ndarray:
    """Run overlapping windows and blend the edges without changing duration."""
    if mixture.ndim != 1 or enrollment.ndim != 1:
        raise ValueError("Mixture and enrollment must be mono")
    if not 0 <= overlap_samples < chunk_samples or chunk_samples < model.n_fft:
        raise ValueError("Invalid chunk or overlap size")
    if len(enrollment) < model.n_fft:
        raise ValueError("Enrollment is shorter than one model STFT window")
    if not len(mixture):
        return np.empty(0, dtype=np.float32)

    hop = chunk_samples - overlap_samples
    total = np.zeros(len(mixture), dtype=np.float32)
    weights = np.zeros(len(mixture), dtype=np.float32)
    sample = torch.from_numpy(enrollment.astype(np.float32, copy=False))[None].to(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(mixture), hop):
            end = min(start + chunk_samples, len(mixture))
            segment = np.zeros(chunk_samples, dtype=np.float32)
            segment[: end - start] = mixture[start:end]
            estimate = model(torch.from_numpy(segment)[None].to(device), sample)
            estimate = estimate.waveform[0, : end - start].cpu().numpy()
            weight = np.ones(end - start, dtype=np.float32)
            fade = min(overlap_samples, len(weight))
            if start > 0 and fade:
                weight[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            if end < len(mixture) and fade:
                weight[-fade:] = np.minimum(
                    weight[-fade:], np.linspace(1, 0, fade, dtype=np.float32)
                )
            total[start:end] += estimate * weight
            weights[start:end] += weight
    return total / np.maximum(weights, 1e-8)


def extract_file(
    checkpoint: str | Path,
    mixture_path: str | Path,
    enrollment_path: str | Path,
    output_path: str | Path,
    device: str = "cpu",
) -> None:
    mixture, rate = sf.read(mixture_path, dtype="float32")
    enrollment, enrollment_rate = sf.read(enrollment_path, dtype="float32")
    if mixture.ndim != 1 or enrollment.ndim != 1:
        raise ValueError("Both input files must be mono")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = state.get("architecture", "band")
    if architecture == "band":
        model = TargetSpeakerExtractor(**state["model_config"])
    elif architecture == "spectral_unet":
        model = SpectralUNetExtractor(**state["model_config"])
    else:
        raise ValueError(f"Unknown checkpoint architecture: {architecture}")
    model.load_state_dict(state["model"])
    model.to(device)
    reduced = _resample(mixture, rate, SAMPLE_RATE)
    voice = _resample(enrollment, enrollment_rate, SAMPLE_RATE)
    output = extract_chunks(model, reduced, voice, device)
    output = _resample(output, SAMPLE_RATE, rate)
    if len(output) < len(mixture):
        output = np.pad(output, (0, len(mixture) - len(output)))
    output = output[: len(mixture)]
    sf.write(output_path, output, rate, subtype="FLOAT")
