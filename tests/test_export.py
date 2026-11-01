import numpy as np

from kezek.annotation import Segment, to_activity
from kezek.export import ExportConfig, export_pseudo_duplex

CFG = ExportConfig(min_segment_s=10.0)


def episode(segments, duration=30.0):
    return to_activity(segments, duration)


def test_two_speakers_with_overlap():
    act, spk = episode([Segment(0, 15, "A"), Segment(14, 28, "B")])
    out = export_pseudo_duplex(act, spk, config=CFG)
    assert out.rejected is None
    assert out.speakers == ("A", "B")
    assert out.overlap.sum() > 0
    assert not (out.channels[0] & out.channels[1]).any()  # overlap silenced in both
    assert not (out.channels[:, out.overlap]).any()
    assert out.segments == [(0, act.shape[1])]


def test_third_speaker_invalidates_frames():
    act, spk = episode(
        [Segment(0, 14, "A"), Segment(14, 27, "B"), Segment(27, 28, "C")], duration=30
    )
    out = export_pseudo_duplex(act, spk, config=CFG)
    assert out.rejected is None
    assert not out.valid[act[spk.index("C")]].any()


def test_rejects_when_top2_share_too_low():
    act, spk = episode([Segment(0, 10, "A"), Segment(10, 20, "B"), Segment(20, 30, "C")])
    out = export_pseudo_duplex(act, spk, config=CFG)
    assert out.rejected is not None and "share" in out.rejected
    assert out.segments == []


def test_low_confidence_splits_segments():
    act, spk = episode([Segment(0, 15, "A"), Segment(15, 30, "B")])
    conf = np.ones(act.shape[1])
    conf[180:190] = 0.1  # ~14.4-15.2 s
    out = export_pseudo_duplex(act, spk, confidence=conf, config=CFG)
    assert out.segments == [(0, 180), (190, act.shape[1])]
    assert out.exported_seconds < 30
