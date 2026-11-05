"""Checks required by the M2S v0.1 specification (section 7) for the two-block Kezek."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "K-M1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "K-R1"))

from m2s_losses import info_nce, mixture_l1, neg_sisdr_active  # noqa: E402
from m2s_model import ConditionedSeparator  # noqa: E402


def test_silence_sisdr_is_finite_and_excluded():
    y = torch.randn(2, 64000)
    s = torch.zeros(2, 64000)
    act = torch.zeros(2, 400, dtype=torch.bool)
    v, valid = neg_sisdr_active(y, s, act)
    assert torch.isfinite(v).all() and not valid.any()


def test_mixture_term_zero_when_outputs_sum_to_mix():
    x = torch.randn(1, 16000)
    a = torch.randn(1, 16000)
    act = torch.ones(1, 100, dtype=torch.bool)
    assert float(mixture_l1(x, a + (x - a), act)) < 1e-6


def test_global_swap_invariant_and_slot_order_matters():
    torch.manual_seed(0)
    r = torch.nn.functional.normalize(torch.randn(3, 2, 16), dim=-1)
    nce = info_nce(r.clone(), r)  # slots equal their references in order
    swapped = info_nce(r.flip(1), r)  # whole-sequence swap of the two slots
    assert torch.allclose(nce[:, 0], swapped[:, 1], atol=1e-5)
    assert (nce[:, 0] < nce[:, 1]).all()


def test_separator_keeps_sample_count():
    sep = ConditionedSeparator().eval()
    for n in (64000, 64011, 128000):
        with torch.no_grad():
            y = sep(torch.randn(1, n), torch.nn.functional.normalize(torch.randn(1, 128), dim=-1))
        assert y.shape[-1] == n
