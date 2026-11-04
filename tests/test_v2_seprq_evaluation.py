"""Checks for amplitude restoration in the direct extraction benchmark."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from evaluate_seprq import align_to_mixture, load_head_checkpoint  # noqa: E402


def test_alignment_restores_target_level_from_peak_normalized_output():
    target = np.array([0.1, 0.0, -0.1, 0.0], dtype=np.float32)
    other = np.array([0.0, 0.05, 0.0, -0.05], dtype=np.float32)

    restored = align_to_mixture(target * 10, target + other)

    assert np.allclose(restored, target, atol=1e-6)


def test_head_checkpoint_rejects_wrong_task(tmp_path):
    class SimplePipeline:
        def __init__(self):
            self.head = torch.nn.Linear(2, 2)

    pipe = SimplePipeline()
    path = tmp_path / "head.pt"
    torch.save({
        "head": pipe.head.state_dict(), "step": 100,
        "config": {"task": "target-speaker-extraction", "upstream": "WavLM_BASE"},
    }, path)

    checksum, step = load_head_checkpoint(
        pipe, path, "target-speaker-extraction", "WavLM_BASE"
    )
    assert len(checksum) == 64
    assert step == 100
    with pytest.raises(ValueError, match="does not match"):
        load_head_checkpoint(pipe, path, "personalized-extraction", "WavLM_BASE")
