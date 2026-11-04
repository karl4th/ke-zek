"""Evaluation emits directly comparable audio and an overlap baseline."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("soundfile")
pytest.importorskip("scipy")

from kezek.metrics import regions, score_extraction  # noqa: E402
from kezek.open_yap import ConversationPair  # noqa: E402
from kezek.tse_v2 import TargetSpeakerExtractor  # noqa: E402
from kezek.tse_v2_unet import SpectralUNetExtractor  # noqa: E402


def test_evaluation_writes_audio_and_scores_overlap(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "V2" / "train.py"
    spec = importlib.util.spec_from_file_location("v2_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    time = np.arange(16_000) / 16_000
    target = torch.from_numpy((0.1 * np.sin(2 * np.pi * 220 * time)).astype("float32"))
    other = torch.from_numpy((0.1 * np.sin(2 * np.pi * 440 * time)).astype("float32"))
    sample = {
        "mixture": target + other,
        "enrollment": target[:8000],
        "other_enrollment": other[:8000],
        "target": target,
        "other": other,
    }
    model = TargetSpeakerExtractor(
        n_fft=64, hop_length=16, bands=4, width=8, blocks=1, enrollment_tokens=4
    )
    scores = module.evaluate(model, [sample], "cpu", tmp_path, demo_count=1)
    assert abs(scores["sisdri_overlap"]) < 1e-3
    assert abs(scores["reverse_sisdri_overlap"]) < 1e-3
    assert scores["enrollment_delta_ratio"] < 1e-3
    assert len(list(tmp_path.glob("*.wav"))) == 3


def test_near_silent_track_is_not_counted_as_speech():
    time = np.arange(16_000) / 16_000
    target = (2e-5 * np.sin(2 * np.pi * 220 * time)).astype("float32")
    other = (0.1 * np.sin(2 * np.pi * 440 * time)).astype("float32")
    reg = regions(target, other, 16_000)
    assert not reg.overlap.any()
    assert reg.absent.all()
    score = score_extraction(target + other, target, other, 16_000)
    assert abs(score.residual_db) < 0.01


def test_split_manifest_keeps_the_original_holdout_when_data_grows(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "V2" / "train.py"
    spec = importlib.util.spec_from_file_location("v2_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    pairs = [
        ConversationPair(
            conversation_id=f"conv_{index}",
            speaker_a_id=f"a_{index}",
            speaker_b_id=f"b_{index}",
            speaker_a_path=tmp_path / f"a_{index}.wav",
            speaker_b_path=tmp_path / f"b_{index}.wav",
            sample_rate=16_000,
            frames=16_000,
        )
        for index in range(4)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"train": ["conv_0"], "val": ["conv_1"], "test": ["conv_2"]}})
    )
    selected = module.select_splits(pairs, manifest, 42)
    assert [[pair.conversation_id for pair in split] for split in selected] == [
        ["conv_0"],
        ["conv_1"],
        ["conv_2"],
    ]


def test_short_overlap_turn_score_detects_preserved_target(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "V2" / "train.py"
    spec = importlib.util.spec_from_file_location("v2_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    (tmp_path / "speaker_a_transcript.json").write_text(
        json.dumps({"words": [{"start": 0.2, "end": 0.55}]})
    )
    (tmp_path / "speaker_b_transcript.json").write_text(
        json.dumps({"words": [{"start": 0.3, "end": 0.7}]})
    )
    sample = {"conversation_dir": str(tmp_path), "target_side": 0, "mixture_start_seconds": 0.0}
    target = np.zeros(16_000, dtype="float32")
    other = np.zeros_like(target)
    target[3200:8800] = 0.05
    other[4800:11200] = 0.05
    baseline = module.short_overlap_turn_score(sample, target + other, target, other)
    recovered = module.short_overlap_turn_score(sample, target, target, other)
    assert baseline is not None and recovered is not None
    assert abs(baseline[0]) < 1e-5
    assert recovered[0] > 30
    assert recovered[1] == 1


def test_speaker_contrastive_loss_trains_voice_profile():
    script = Path(__file__).resolve().parents[1] / "experiments" / "V2" / "train.py"
    spec = importlib.util.spec_from_file_location("v2_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = SpectralUNetExtractor(n_fft=64, hop_length=16)
    target = torch.randn(2, 1024) * 0.1
    other = torch.randn(2, 1024) * 0.1
    enrollment = torch.randn(2, 640) * 0.1
    loss, count = module.speaker_contrastive_loss(model, enrollment, target, other)
    assert count == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert model.profile[1].weight.grad is not None
    assert model.profile[1].weight.grad.abs().sum() > 0


def test_new_run_can_initialize_from_matching_model_checkpoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "V2" / "train.py"
    spec = importlib.util.spec_from_file_location("v2_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = {"n_fft": 64, "hop_length": 16, "bands": 4, "width": 8, "blocks": 1}
    source = TargetSpeakerExtractor(**config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"architecture": "band", "model_config": config, "model": source.state_dict()},
        checkpoint,
    )
    loaded = TargetSpeakerExtractor(**config)
    module.load_initial_model(loaded, checkpoint, "band", config)
    for expected, actual in zip(source.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    with pytest.raises(ValueError, match="do not match"):
        module.load_initial_model(loaded, checkpoint, "spectral_unet", config)
