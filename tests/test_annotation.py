import numpy as np
import pytest

from kezek.annotation import Segment, read_rttm, read_uem, to_activity, to_segments, write_rttm
from kezek.frames import MIMI_FRAME_RATE_HZ, num_frames, time_to_frame


def test_frame_grid():
    assert num_frames(1.0) == 13  # 12.5 frames -> 13 to cover the full second
    assert num_frames(0.8) == 10
    assert time_to_frame(0.08) == 1
    assert time_to_frame(0.0799) == 0


def test_rttm_roundtrip(tmp_path):
    ann = {
        "ep1": [Segment(0.0, 1.5, "A"), Segment(1.2, 3.0, "B")],
        "ep2": [Segment(0.5, 0.9, "X")],
    }
    path = tmp_path / "x.rttm"
    write_rttm(path, ann)
    back = read_rttm(path)
    assert set(back) == {"ep1", "ep2"}
    for k in ann:
        for a, b in zip(ann[k], back[k], strict=True):
            assert a.speaker == b.speaker
            assert a.start == pytest.approx(b.start, abs=1e-3)
            assert a.end == pytest.approx(b.end, abs=1e-3)


def test_read_uem(tmp_path):
    path = tmp_path / "x.uem"
    path.write_text("ep1 1 0.000 60.000\nep1 1 70.0 80.0\n")
    assert read_uem(path) == {"ep1": [(0.0, 60.0), (70.0, 80.0)]}


def test_activity_roundtrip_on_frame_grid():
    rate = MIMI_FRAME_RATE_HZ
    segs = [Segment(0.0, 0.8, "A"), Segment(0.4, 1.6, "B")]
    act, spk = to_activity(segs, duration_s=2.0, frame_rate=rate)
    assert spk == ["A", "B"]
    assert act.shape == (2, 25)
    assert act[0].sum() == 10 and act[1].sum() == 15
    assert act[:, 5:10].all()  # overlap 0.4-0.8 s
    back = to_segments(act, spk, rate)
    assert back == sorted(segs)


def test_activity_rejects_unknown_speaker():
    with pytest.raises(KeyError):
        to_activity([Segment(0, 1, "C")], speakers=["A"])


def test_segment_validation():
    with pytest.raises(ValueError):
        Segment(2.0, 1.0, "A")
    assert np.isclose(Segment(1.0, 2.5, "A").duration, 1.5)
