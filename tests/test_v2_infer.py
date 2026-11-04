"""Chunked inference keeps the full input duration and joins windows smoothly."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

from kezek.tse_v2 import TargetSpeakerExtractor  # noqa: E402
from kezek.tse_v2_infer import extract_chunks, extract_file  # noqa: E402


def small_model():
    return TargetSpeakerExtractor(
        n_fft=64, hop_length=16, bands=4, width=8, blocks=1, enrollment_tokens=4
    )


def test_chunked_initial_model_preserves_nonmultiple_length():
    rng = np.random.default_rng(5)
    mixture = (rng.standard_normal(2501) * 0.1).astype("float32")
    enrollment = (rng.standard_normal(640) * 0.1).astype("float32")
    output = extract_chunks(
        small_model(), mixture, enrollment, "cpu", chunk_samples=1024, overlap_samples=256
    )
    np.testing.assert_allclose(output, mixture, atol=2e-5)


def test_file_inference_restores_original_sample_rate_and_length(tmp_path):
    model = small_model()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {
                "n_fft": 64,
                "hop_length": 16,
                "bands": 4,
                "width": 8,
                "blocks": 1,
                "enrollment_tokens": 4,
            },
        },
        checkpoint,
    )
    rate = 48_000
    time = np.arange(4801) / rate
    mixture = (0.1 * np.sin(2 * np.pi * 220 * time)).astype("float32")
    source, voice, result = (tmp_path / name for name in ("mix.wav", "voice.wav", "out.wav"))
    sf.write(source, mixture, rate)
    sf.write(voice, mixture, rate)
    extract_file(checkpoint, source, voice, result)
    output, output_rate = sf.read(result)
    assert output_rate == rate
    assert len(output) == len(mixture)
