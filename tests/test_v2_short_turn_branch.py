"""Verify that the temporal correction starts from the fixed extraction mask."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from short_turn_branch import TemporalMaskCorrection  # noqa: E402


def test_correction_preserves_base_at_initialization_and_trains_only_new_branch():
    base = torch.nn.Linear(16, 8)
    inputs = torch.randn(2, 31, 16)
    expected = base(inputs).detach()
    branch = TemporalMaskCorrection(base, width=12)

    initial = branch(inputs)
    assert initial.shape == (2, 31, 8)
    assert torch.allclose(initial, expected)

    initial.square().mean().backward()
    assert base.weight.grad is None
    assert branch.delta.weight.grad is not None
    assert branch.delta.weight.grad.abs().sum() > 0


def test_correction_rejects_wrong_feature_shape():
    branch = TemporalMaskCorrection(torch.nn.Linear(16, 8))
    with pytest.raises(ValueError, match="hidden"):
        branch(torch.randn(2, 31, 15))
