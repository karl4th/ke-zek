"""Real-pair window construction and training-loss checks."""

import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from kezek.open_yap import scan_conversations  # noqa: E402
from kezek.open_yap_training import OpenYapWindows, _separate_start  # noqa: E402
from kezek.tse_v2 import TargetSpeakerExtractor  # noqa: E402
from kezek.tse_v2_loss import extraction_loss  # noqa: E402


def test_enrollment_never_overlaps_mixture():
    rng = np.random.default_rng(3)
    for crop_start in (0, 20, 50, 80):
        for _ in range(50):
            start = _separate_start(rng, 100, crop_start, 10, 5, 2)
            assert start + 5 + 2 <= crop_start or start >= crop_start + 10 + 2


def test_window_preserves_aligned_pair_and_resamples(tmp_path):
    rate, seconds = 48_000, 20
    directory = tmp_path / "conversations" / "conv_demo"
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(
        json.dumps({"conversation_id": "demo", "speaker_a_id": "a", "speaker_b_id": "b"})
    )
    time = np.arange(rate * seconds) / rate
    sf.write(directory / "speaker_a.wav", 0.1 * np.sin(2 * np.pi * 220 * time), rate)
    sf.write(directory / "speaker_b.wav", 0.05 * np.sin(2 * np.pi * 440 * time), rate)
    pairs = scan_conversations(tmp_path)
    dataset = OpenYapWindows(pairs, count=2, seed=2, mixture_seconds=2, enrollment_seconds=3)
    first, second = dataset[0], dataset[1]
    for item in (first, second):
        assert item["mixture"].shape == (32_000,)
        assert item["enrollment"].shape == (48_000,)
        torch.testing.assert_close(item["mixture"], item["target"] + item["other"])
        assert item["conversation_id"] == "demo"
        assert item["conversation_dir"] == str(directory)
        assert item["mixture_start_seconds"] >= 0
        assert item["enrollment_start_seconds"] >= 0
    assert not torch.allclose(first["target"], second["target"], atol=1e-3)


def test_loss_is_finite_and_reaches_model():
    model = TargetSpeakerExtractor(
        n_fft=64, hop_length=16, bands=4, width=8, blocks=1, enrollment_tokens=4
    )
    target = torch.zeros(1, 3200)
    other = torch.zeros_like(target)
    target[:, :1600] = torch.randn(1, 1600) * 0.1
    other[:, 800:] = torch.randn(1, 2400) * 0.1
    output = model(target + other, torch.randn(1, 640) * 0.1)
    loss, parts = extraction_loss(output, target, other)
    assert torch.isfinite(loss)
    assert set(parts) == {"active_snr_db", "overlap_snr_db", "absent_residual_db", "activity_bce"}
    loss.backward()
    assert model.band_outputs[0].weight.grad is not None


def test_overlap_sampling_centers_training_crops_on_words(tmp_path):
    directory = tmp_path / "conversations" / "conv_demo"
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(
        json.dumps({"conversation_id": "demo", "speaker_a_id": "a", "speaker_b_id": "b"})
    )
    for side, start, end in (("a", 10.0, 11.0), ("b", 10.5, 11.5)):
        sf.write(directory / f"speaker_{side}.wav", np.zeros(20 * 16_000), 16_000)
        (directory / f"speaker_{side}_transcript.json").write_text(
            json.dumps({"words": [{"start": start, "end": end}]})
        )
    dataset = OpenYapWindows(
        scan_conversations(tmp_path),
        count=20,
        seed=3,
        mixture_seconds=2,
        enrollment_seconds=3,
        overlap_sampling=1.0,
    )
    for sample in dataset:
        assert sample["mixture_start_seconds"] <= 10.75
        assert sample["mixture_start_seconds"] + 2 >= 10.75
