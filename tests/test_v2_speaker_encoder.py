"""Speaker adaptation rewards the enrolled voice over the other voice."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
AF = pytest.importorskip("torchaudio.functional")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_speaker_encoder import separated_pairs, speaker_pair_loss  # noqa: E402


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


def test_frozen_separator_pairing_follows_clean_target_when_outputs_swap():
    time = torch.arange(16_000, dtype=torch.float32) / 16_000
    target = (0.1 * torch.sin(2 * torch.pi * 220 * time))[None]
    other = (0.1 * torch.sin(2 * torch.pi * 440 * time))[None]

    class ReversedSeparator:
        def separate_batch(self, _mixture):
            return 5 * torch.stack((
                AF.resample(other, 16_000, 8_000),
                AF.resample(target, 16_000, 8_000),
            ), dim=-1)

    positive, negative = separated_pairs(ReversedSeparator(), target + other, target, other)
    assert positive.shape == target.shape
    assert torch.mean((positive - target) ** 2) < torch.mean((negative - target) ** 2)
