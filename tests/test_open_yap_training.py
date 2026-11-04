"""Real-pair window construction and training-loss checks."""

import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from kezek.open_yap import scan_conversations  # noqa: E402
from kezek.open_yap_training import CrossPairWindows, OpenYapWindows, _separate_start  # noqa: E402
from kezek.tse_v2 import TargetSpeakerExtractor  # noqa: E402
from kezek.tse_v2_loss import (  # noqa: E402
    extraction_loss,
    mask_supervision_loss,
    source_discriminative_loss,
)


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
        assert item["other_enrollment"].shape == (48_000,)
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


def test_power_selection_prefers_stronger_clean_enrollment(tmp_path):
    directory = tmp_path / "conversations" / "conv_demo"
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(
        json.dumps({"conversation_id": "demo", "speaker_a_id": "a", "speaker_b_id": "b"})
    )
    rate, seconds = 16_000, 20
    time = np.arange(rate * seconds) / rate
    tone = np.sin(2 * np.pi * 220 * time)
    sf.write(directory / "speaker_a.wav", tone * np.where(time < 10, 0.01, 0.1), rate)
    sf.write(directory / "speaker_b.wav", tone * np.where(time < 10, 0.00001, 0.1), rate)
    pairs = scan_conversations(tmp_path)
    ratio = OpenYapWindows(pairs, 30, seed=7, mixture_seconds=2, enrollment_seconds=3)
    power = OpenYapWindows(
        pairs, 30, seed=7, mixture_seconds=2, enrollment_seconds=3,
        enrollment_selection="power",
    )
    improvements = [
        float(power[index]["enrollment"].square().mean())
        - float(ratio[index]["enrollment"].square().mean())
        for index in range(30)
    ]
    assert all(value >= -1e-8 for value in improvements)
    assert any(value > 1e-4 for value in improvements)


def test_discriminative_loss_prefers_target_on_overlap():
    time = torch.arange(3200, dtype=torch.float32) / 16_000
    target = (0.1 * torch.sin(2 * torch.pi * 220 * time))[None]
    other = (0.1 * torch.sin(2 * torch.pi * 440 * time))[None]
    correct, count = source_discriminative_loss(target, target, other)
    wrong, _ = source_discriminative_loss(other, target, other)
    assert count == 1
    assert correct < wrong
    estimate = (target + other).requires_grad_()
    mixed, _ = source_discriminative_loss(estimate, target, other)
    mixed.backward()
    assert estimate.grad is not None
    assert estimate.grad.abs().sum() > 0


def test_cross_pair_windows_use_different_speakers_and_keep_sum(tmp_path):
    rate, seconds = 16_000, 20
    time = np.arange(rate * seconds) / rate
    for index in range(2):
        directory = tmp_path / "conversations" / f"conv_{index}"
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(
            json.dumps({
                "conversation_id": f"conv_{index}",
                "speaker_a_id": f"a_{index}",
                "speaker_b_id": f"b_{index}",
            })
        )
        sf.write(directory / "speaker_a.wav", 0.1 * np.sin(2 * np.pi * 220 * time), rate)
        sf.write(directory / "speaker_b.wav", 0.1 * np.sin(2 * np.pi * 330 * time), rate)
    natural = OpenYapWindows(
        scan_conversations(tmp_path), 20, seed=3, mixture_seconds=2, enrollment_seconds=3
    )
    mixed = CrossPairWindows(natural, probability=1.0, seed=9)
    for index in range(10):
        sample = mixed[index]
        assert sample["cross_pair"]
        assert sample["target_speaker_id"] != sample["other_speaker_id"]
        torch.testing.assert_close(sample["mixture"], sample["target"] + sample["other"])


def test_mask_supervision_distinguishes_target_from_other():
    model = TargetSpeakerExtractor(
        n_fft=64, hop_length=16, bands=4, width=8, blocks=1, enrollment_tokens=4
    )
    target = torch.randn(1, 1024) * 0.1
    other = torch.zeros_like(target)
    output = model(target, torch.randn(1, 640) * 0.1)
    correct = mask_supervision_loss(output, target, other, 64, 16)
    wrong = mask_supervision_loss(output, other, target, 64, 16)
    assert correct < 1e-6
    assert wrong > 0.9
    wrong.backward()
    assert model.band_outputs[0].weight.grad is not None
