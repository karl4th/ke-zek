"""Shape, routing, and gradient checks for the direct target speaker extractor."""

import pytest

torch = pytest.importorskip("torch")

from kezek.tse_v2 import TargetSpeakerExtractor  # noqa: E402


def small_model():
    return TargetSpeakerExtractor(
        n_fft=64, hop_length=16, bands=4, width=8, blocks=1, enrollment_tokens=4
    )


def test_initial_output_keeps_alignment_and_signal():
    model = small_model()
    mixture = torch.randn(2, 1031) * 0.1
    enrollment = torch.randn(2, 645) * 0.1
    output = model(mixture, enrollment)
    assert output.waveform.shape == mixture.shape
    assert output.activity_logits.shape[0] == len(mixture)
    assert output.activity_logits.shape[1] == mixture.shape[1] // model.hop_length + 1
    torch.testing.assert_close(output.waveform, mixture, atol=1e-5, rtol=1e-5)


def test_enrollment_changes_output_and_gradients_reach_conditioning():
    torch.manual_seed(4)
    model = small_model()
    for head in model.band_outputs:
        torch.nn.init.normal_(head.weight, std=0.01)
    mixture = torch.randn(1, 1024) * 0.1
    first = torch.randn(1, 640) * 0.1
    second = torch.randn(1, 640) * 0.1
    a = model(mixture, first).waveform
    b = model(mixture, second).waveform
    assert not torch.allclose(a, b, atol=1e-7)
    a.square().mean().backward()
    assert model.voice_fusion.weight.grad is not None
    assert model.voice_fusion.weight.grad.abs().sum() > 0


def test_band_concat_conditioning_preserves_band_features_for_gradient():
    model = TargetSpeakerExtractor(
        n_fft=64, hop_length=16, bands=4, width=8, blocks=1,
        enrollment_tokens=4, conditioning="band_concat",
    )
    for head in model.band_outputs:
        torch.nn.init.normal_(head.weight, std=0.01)
    mixture = torch.randn(1, 1024) * 0.1
    enrollment = torch.randn(1, 640) * 0.1
    output = model(mixture, enrollment).waveform
    assert output.shape == mixture.shape
    output.square().mean().backward()
    assert model.voice_band_projection[1].weight.grad is not None
    assert model.voice_band_projection[1].weight.grad.abs().sum() > 0


def test_rejects_incompatible_waveforms():
    model = small_model()
    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.zeros(2, 1024), torch.zeros(1, 640))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1024), torch.zeros(640))
    with pytest.raises(ValueError, match="STFT window"):
        model(torch.zeros(1, 1024), torch.zeros(1, 32))
