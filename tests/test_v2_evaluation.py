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
