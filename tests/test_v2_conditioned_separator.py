"""Checks for enrollment-conditioned separation and source ordering."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kezek.conditioned_separator import ConditionedSeparator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_conditioned_separator import PairedTargetWindows, direct_loss  # noqa: E402


class _Encoder(torch.nn.Module):
    def forward(self, mixture):
        return mixture[:, None]


class _Masknet(torch.nn.Module):
    def forward(self, encoded):
        return torch.stack((encoded.sigmoid(), (-encoded).sigmoid()))


class _Decoder(torch.nn.Module):
    def forward(self, encoded):
        return encoded[:, 0]


def test_conditioned_separator_matches_unconditioned_initialization():
    model = ConditionedSeparator(_Encoder(), _Masknet(), _Decoder(), channels=1, voice_dim=2)
    mixture = torch.tensor([[0.2, -0.4, 0.6]])
    voice = torch.tensor([[1.0, 0.0]])

    estimated = model(mixture, voice)
    expected = torch.stack((
        mixture * mixture.sigmoid(), mixture * (-mixture).sigmoid(),
    ), dim=-1)

    assert estimated.shape == (1, 3, 2)
    assert torch.allclose(estimated, expected)
    estimated[..., 0].sum().backward()
    assert model.condition.weight.grad.abs().sum() > 0


def test_conditioned_separator_rejects_mismatched_voice_batch():
    model = ConditionedSeparator(_Encoder(), _Masknet(), _Decoder(), channels=1, voice_dim=2)
    with pytest.raises(ValueError, match="batch shapes"):
        model(torch.zeros(2, 64), torch.zeros(1, 2))


def test_paired_targets_reverse_the_same_mixture():
    class SingleWindow:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {
                "mixture": torch.tensor([0.3, 0.7]),
                "target": torch.tensor([0.1, 0.2]),
                "other": torch.tensor([0.2, 0.5]),
                "enrollment": torch.tensor([0.1]),
                "other_enrollment": torch.tensor([0.2]),
                "target_side": 0,
                "target_speaker_id": "a",
                "other_speaker_id": "b",
            }

    paired = PairedTargetWindows(SingleWindow())
    first, second = paired[0], paired[1]

    assert len(paired) == 2
    assert torch.equal(first["mixture"], second["mixture"])
    assert torch.equal(first["target"], second["other"])
    assert torch.equal(first["enrollment"], second["other_enrollment"])
    assert second["target_side"] == 1


def test_direct_loss_prefers_requested_source_first():
    time = torch.arange(8_000) / 8_000
    target = (0.1 * torch.sin(2 * torch.pi * 220 * time))[None]
    other = (0.1 * torch.sin(2 * torch.pi * 440 * time))[None]
    correctly_ordered = torch.stack((target, other), dim=-1)
    reversed_order = correctly_ordered.flip(-1)

    good, _ = direct_loss(correctly_ordered, target, other, target + other)
    wrong, _ = direct_loss(reversed_order, target, other, target + other)

    assert good < wrong
