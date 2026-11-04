"""Archive indexing checks for paired conversation tracks."""

import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from kezek.open_yap import (  # noqa: E402
    check_speaker_disjoint,
    scan_conversations,
    split_conversations,
)


def _pair(root, name, speakers=("a1", "b1"), frames=1600):
    directory = root / "conversations" / f"conv_{name}"
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {"conversation_id": name, "speaker_a_id": speakers[0], "speaker_b_id": speakers[1]}
        )
    )
    for side in "ab":
        sf.write(directory / f"speaker_{side}.wav", np.zeros(frames), 16000)
    return directory


def test_scan_and_speaker_disjointness(tmp_path):
    _pair(tmp_path, "first", ("a1", "b1"))
    _pair(tmp_path, "second", ("a1", "b2"))
    pairs = scan_conversations(tmp_path)
    assert len(pairs) == 2
    assert pairs[0].frames == 1600 and pairs[0].sample_rate == 16000
    check_speaker_disjoint(pairs)
    with pytest.raises(ValueError, match="multiple splits"):
        check_speaker_disjoint([pairs[0]], [pairs[1]])


def test_rejects_misaligned_tracks(tmp_path):
    directory = _pair(tmp_path, "first")
    sf.write(directory / "speaker_b.wav", np.zeros(800), 16000)
    with pytest.raises(ValueError, match="sample-aligned"):
        scan_conversations(tmp_path)


def test_split_keeps_connected_speakers_together(tmp_path):
    _pair(tmp_path, "one", ("a1", "b1"))
    _pair(tmp_path, "two", ("a1", "b2"))
    _pair(tmp_path, "three", ("a3", "b3"))
    _pair(tmp_path, "four", ("a4", "b4"))
    _pair(tmp_path, "five", ("a5", "b5"))
    splits = split_conversations(scan_conversations(tmp_path), seed=7)
    assert sum(map(len, splits)) == 5
    check_speaker_disjoint(*splits)
    assert all(splits)
    locations = {
        pair.conversation_id: index for index, split in enumerate(splits) for pair in split
    }
    assert locations["one"] == locations["two"]
