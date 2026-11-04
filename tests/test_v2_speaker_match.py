"""Energy trimming keeps enough speech for a speaker embedding."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from evaluate_speaker_clean import trim_by_energy  # noqa: E402


def test_energy_trim_removes_long_silence_and_retains_speech():
    audio = np.concatenate(
        (np.zeros(16_000), np.full(16_000, 0.1), np.zeros(16_000))
    ).astype("float32")
    trimmed = trim_by_energy(audio)
    assert len(trimmed) == 16_000
    assert np.all(trimmed == 0.1)
