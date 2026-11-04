"""The spectral alternative accepts enrollment and preserves audio alignment."""

import pytest

torch = pytest.importorskip("torch")

from kezek.tse_v2_unet import SpectralUNetExtractor  # noqa: E402


def test_initial_unet_passes_mixture_and_routes_enrollment_gradient():
    model = SpectralUNetExtractor(n_fft=64, hop_length=16)
    mixture = torch.randn(1, 1024) * 0.1
    enrollment = torch.randn(1, 640) * 0.1
    output = model(mixture, enrollment)
    assert output.waveform.shape == mixture.shape
    torch.testing.assert_close(output.waveform, mixture, atol=1e-5, rtol=1e-5)
    torch.nn.init.normal_(model.mask.weight, std=0.01)
    changed = model(mixture, enrollment).waveform
    changed.square().mean().backward()
    assert model.profile[1].weight.grad is not None
    assert model.profile[1].weight.grad.abs().sum() > 0


def test_unet_rejects_mismatched_batches():
    model = SpectralUNetExtractor(n_fft=64, hop_length=16)
    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.zeros(2, 1024), torch.zeros(1, 640))
