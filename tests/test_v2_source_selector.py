"""The learned selector respects source order and scores the chosen output."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_source_selector import evaluate_selector, feature_vector  # noqa: E402


def test_pair_features_reverse_when_sources_swap():
    enrollment = torch.tensor([1.0, 0.0])
    first = torch.tensor([0.9, 0.1])
    second = torch.tensor([0.1, 0.9])
    direct = feature_vector(enrollment, first, second, 0.2, 0.1)
    swapped = feature_vector(enrollment, second, first, 0.1, 0.2)
    torch.testing.assert_close(direct, -swapped)


def test_selector_evaluation_uses_chosen_source_metrics():
    rows = [{
        "oracle": 1,
        "overlap_s": 0.5,
        "source_sisdri_overlap": [-4.0, 3.0],
        "source_short_overlap_improvement_db": [-2.0, 1.0],
    }]
    result = evaluate_selector(torch.tensor([-0.5]), rows)
    assert result["assignment_accuracy"] == 1.0
    assert result["sisdri_overlap"] == 3.0
    assert result["short_overlap_improvement_db"] == 1.0
