"""Speaker adaptation rewards the enrolled voice over the other voice."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_speaker_encoder import speaker_pair_loss  # noqa: E402


class SimpleEncoder:
    def encode_batch(self, waveforms):
        return waveforms[:, None, :2]


def test_pair_loss_prefers_enrolled_source():
    enrollment = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    same = torch.tensor([[2.0, 0.0, -2.0, 0.0]])
    different = torch.tensor([[0.0, 1.0, 0.0, -1.0]])
    good, metrics = speaker_pair_loss(SimpleEncoder(), enrollment, same, different)
    reversed_loss, _ = speaker_pair_loss(SimpleEncoder(), enrollment, different, same)
    assert good < reversed_loss
    assert metrics["pair_accuracy"] == 1.0
