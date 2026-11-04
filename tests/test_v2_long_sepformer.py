"""Conversation-level speaker assignment uses surrounding evidence when helpful."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from evaluate_long_sepformer import choose_sources  # noqa: E402


def test_continuity_recovers_one_ambiguous_chunk():
    emissions = np.array([[0.8, 0.1], [0.4, 0.5], [0.8, 0.1]])
    continuity = np.zeros((2, 2, 2))
    continuity[:, 0, 0] = 0.9
    continuity[:, 1, 1] = 0.9
    assert choose_sources(emissions, continuity, 0.0) == [0, 1, 0]
    assert choose_sources(emissions, continuity, 0.25) == [0, 0, 0]
