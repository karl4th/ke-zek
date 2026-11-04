"""Permutation training keeps both speakers and does not depend on output order."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_sepformer import align_sources_to_mixture, permutation_loss  # noqa: E402


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


def test_common_gain_restores_source_scale_without_changing_sum():
    time = torch.arange(3200, dtype=torch.float32) / 8000
    first = (0.1 * torch.sin(2 * torch.pi * 220 * time))[None]
    second = (0.1 * torch.sin(2 * torch.pi * 440 * time))[None]
    mixture = first + second
    estimates = 20 * torch.stack((first, second), dim=-1)
    aligned, gain = align_sources_to_mixture(estimates, mixture)
    torch.testing.assert_close(gain, torch.tensor([0.05]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(aligned.sum(dim=-1), mixture, atol=1e-5, rtol=1e-5)
    loss, _ = permutation_loss(aligned, first, second, mixture)
    assert loss < 0
