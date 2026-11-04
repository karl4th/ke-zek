"""Checks for direct target extraction training objectives."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_seprq_tse import align_batch, target_loss  # noqa: E402


def test_target_loss_prefers_requested_voice():
    time = torch.arange(16_000) / 16_000
    target = (0.1 * torch.sin(2 * torch.pi * 220 * time))[None]
    other = (0.1 * torch.sin(2 * torch.pi * 440 * time))[None]
    mixture = target + other

    good, _ = target_loss(target, target, other, mixture)
    wrong, _ = target_loss(other, target, other, mixture)

    assert good < wrong
    assert torch.allclose(align_batch(target * 10, mixture), target, atol=1e-5)


def test_target_loss_penalizes_distractor_when_target_is_absent():
    time = torch.arange(16_000) / 16_000
    other = (0.1 * torch.sin(2 * torch.pi * 440 * time))[None]
    target = torch.zeros_like(other)

    silence, _ = target_loss(target, target, other, other)
    leaked, _ = target_loss(other, target, other, other)

    assert silence < leaked
