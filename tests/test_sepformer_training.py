"""Permutation training keeps both speakers and does not depend on output order."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_sepformer import permutation_loss  # noqa: E402


def test_permutation_loss_prefers_both_sources_and_ignores_output_order():
    time = torch.arange(3200, dtype=torch.float32) / 8000
    first = (0.1 * torch.sin(2 * torch.pi * 220 * time))[None]
    second = (0.1 * torch.sin(2 * torch.pi * 440 * time))[None]
    mixture = first + second
    direct = torch.stack((first, second), dim=-1)
    reversed_outputs = torch.stack((second, first), dim=-1)
    silenced = torch.zeros_like(direct)
    best, _ = permutation_loss(direct, first, second, mixture)
    swapped, _ = permutation_loss(reversed_outputs, first, second, mixture)
    collapsed, _ = permutation_loss(silenced, first, second, mixture)
    torch.testing.assert_close(best, swapped)
    assert best < collapsed
