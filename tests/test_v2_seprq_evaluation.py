"""Checks for amplitude restoration in the direct extraction benchmark."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from evaluate_seprq import align_to_mixture  # noqa: E402


def test_alignment_restores_target_level_from_peak_normalized_output():
    target = np.array([0.1, 0.0, -0.1, 0.0], dtype=np.float32)
    other = np.array([0.0, 0.05, 0.0, -0.05], dtype=np.float32)

    restored = align_to_mixture(target * 10, target + other)

    assert np.allclose(restored, target, atol=1e-6)
