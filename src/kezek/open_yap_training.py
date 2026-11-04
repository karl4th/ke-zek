"""Deterministic target-speaker windows from aligned Open Yap conversations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from kezek.open_yap import ConversationPair

SAMPLE_RATE = 16_000


def _overlap_centers(pair: ConversationPair) -> np.ndarray:
    """Find word-level overlap points for sampling conversational interruptions."""
    directory = pair.speaker_a_path.parent
    tracks = []
    for side in "ab":
        words = json.loads((directory / f"speaker_{side}_transcript.json").read_text())["words"]
        tracks.append(sorted((float(word["start"]), float(word["end"])) for word in words))
    first, second = tracks
    i = j = 0
    centers = []
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end - start >= 0.04:
            centers.append(round((start + end) * 0.5 * pair.sample_rate))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return np.asarray(centers, dtype=np.int64)


def _read_window(path: Path, start: int, frames: int, sample_rate: int) -> np.ndarray:
    with sf.SoundFile(path) as audio:
        audio.seek(start)
        samples = audio.read(frames, dtype="float32", always_2d=False)
    if len(samples) != frames:
        raise ValueError(f"Short read from {path}: {len(samples)} of {frames} frames")
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, sample_rate // divisor).astype(
            np.float32
        )
    return samples


def _separate_start(
    rng: np.random.Generator,
    total: int,
    crop_start: int,
    crop_frames: int,
    sample_frames: int,
    gap_frames: int,
) -> int:
    last = total - sample_frames
    before = max(0, crop_start - gap_frames - sample_frames + 1)
    after_start = crop_start + crop_frames + gap_frames
    after = max(0, last - after_start + 1)
    if before + after == 0:
        raise ValueError("Conversation is too short for separate mixture and enrollment")
    pick = int(rng.integers(before + after))
    return pick if pick < before else after_start + pick - before


class OpenYapWindows(torch.utils.data.Dataset):
    """Read short windows on demand, without loading whole conversations into memory."""

    def __init__(
        self,
        pairs: list[ConversationPair],
        count: int,
        seed: int,
        mixture_seconds: float = 4.0,
        enrollment_seconds: float = 6.0,
        gap_seconds: float = 2.0,
        enrollment_candidates: int = 4,
        overlap_sampling: float = 0.0,
    ) -> None:
        if count < 1 or mixture_seconds <= 0 or enrollment_seconds <= 0:
            raise ValueError("Count and window durations must be positive")
        if gap_seconds < 0 or enrollment_candidates < 1:
            raise ValueError("Gap must be nonnegative and enrollment candidates positive")
        if not 0 <= overlap_sampling <= 1:
            raise ValueError("Overlap sampling must be between zero and one")
        self.count, self.seed = count, seed
        self.mixture_seconds = mixture_seconds
        self.enrollment_seconds = enrollment_seconds
        self.gap_seconds = gap_seconds
        self.enrollment_candidates = enrollment_candidates
        self.overlap_sampling = overlap_sampling
        self.pairs = [
            pair
            for pair in pairs
            if pair.frames
            >= round((mixture_seconds + enrollment_seconds + gap_seconds) * pair.sample_rate)
        ]
        if not self.pairs:
            raise ValueError("No conversations are long enough for the requested windows")
        self.overlap_centers = (
            [_overlap_centers(pair) for pair in self.pairs] if overlap_sampling else []
        )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        rng = np.random.default_rng([self.seed, index])
        pair_index = int(rng.integers(len(self.pairs)))
        pair = self.pairs[pair_index]
        rate = pair.sample_rate
        crop_frames = round(self.mixture_seconds * rate)
        sample_frames = round(self.enrollment_seconds * rate)
        gap_frames = round(self.gap_seconds * rate)
        crop_start = int(rng.integers(pair.frames - crop_frames + 1))
        if self.overlap_sampling and rng.random() < self.overlap_sampling:
            centers = self.overlap_centers[pair_index]
            if len(centers):
                center = int(centers[int(rng.integers(len(centers)))])
                offset = int(rng.integers(crop_frames))
                crop_start = max(0, min(pair.frames - crop_frames, center - offset))
        for _ in range(8):
            if (
                crop_start >= sample_frames + gap_frames
                or crop_start + crop_frames + gap_frames + sample_frames <= pair.frames
            ):
                break
            crop_start = int(rng.integers(pair.frames - crop_frames + 1))
        else:
            crop_start = 0
        paths = (pair.speaker_a_path, pair.speaker_b_path)
        target_side = index % 2
        target_path, other_path = paths[target_side], paths[1 - target_side]

        target = _read_window(target_path, crop_start, crop_frames, rate)
        other = _read_window(other_path, crop_start, crop_frames, rate)
        best_score = -float("inf")
        enrollment = None
        enrollment_start = None
        for _ in range(self.enrollment_candidates):
            sample_start = _separate_start(
                rng, pair.frames, crop_start, crop_frames, sample_frames, gap_frames
            )
            candidate = _read_window(target_path, sample_start, sample_frames, rate)
            other_sample = _read_window(other_path, sample_start, sample_frames, rate)
            target_power = float(np.mean(candidate**2))
            other_power = float(np.mean(other_sample**2))
            score = target_power / (other_power + 1e-6)
            if target_power < 1e-6:
                score = -1.0
            if score > best_score:
                best_score, enrollment = score, candidate
                enrollment_start = sample_start
        assert enrollment is not None and enrollment_start is not None

        mixture = target + other
        gain = min(1.0, 0.95 / max(float(np.max(np.abs(mixture))), 1e-6))
        target, other, mixture = target * gain, other * gain, mixture * gain
        return {
            "mixture": torch.from_numpy(mixture.copy()),
            "enrollment": torch.from_numpy(enrollment.copy()),
            "target": torch.from_numpy(target.copy()),
            "other": torch.from_numpy(other.copy()),
            "conversation_id": pair.conversation_id,
            "conversation_dir": str(pair.speaker_a_path.parent),
            "target_side": target_side,
            "mixture_start_seconds": crop_start / rate,
            "enrollment_start_seconds": enrollment_start / rate,
        }
