import numpy as np
import pytest

from kezek.annotation import Segment, to_activity, write_rttm
from kezek.cli import evaluate
from kezek.metrics import diarization_error, duplex_counts
from kezek.metrics.duplex import bridge_gaps

RATE = 100.0


def act(segments, duration=10.0, speakers=None):
    return to_activity(segments, duration, RATE, speakers)[0]


REF = [Segment(0, 4, "A"), Segment(4, 8, "B")]


def test_perfect_hypothesis_with_relabelled_speakers():
    hyp = [Segment(0, 4, "spk1"), Segment(4, 8, "spk0")]
    d = diarization_error(act(REF), act(hyp), RATE)
    assert d.der == 0.0
    s = duplex_counts(act(REF), act(hyp), RATE).summary()
    assert s["boundary_median_ms"] == 0.0
    assert s["leakage"] == 0.0
    assert s["vad_f1"] == 1.0


def test_missed_false_alarm_confusion():
    # A spoken 0-4, B 4-8. Hyp: A 0-3 (1 s missed), B 4-9 (1 s false alarm),
    # plus speaker C on 6-8 overlapping B (2 s false alarm).
    hyp = [Segment(0, 3, "A"), Segment(4, 9, "B"), Segment(6, 8, "C")]
    d = diarization_error(act(REF), act(hyp), RATE)
    assert d.total == pytest.approx(8.0)
    assert d.missed == pytest.approx(1.0)
    assert d.false_alarm == pytest.approx(3.0)
    assert d.confusion == pytest.approx(0.0)
    assert d.der == pytest.approx(0.5)

    # All speech attributed to one speaker: half of it is confusion.
    d = diarization_error(act(REF), act([Segment(0, 8, "X")]), RATE)
    assert d.confusion == pytest.approx(4.0)
    assert d.der == pytest.approx(0.5)


def test_collar_and_uem():
    hyp = [Segment(0, 4.2, "A"), Segment(4.2, 8, "B")]
    assert diarization_error(act(REF), act(hyp), RATE).der > 0
    assert diarization_error(act(REF), act(hyp), RATE, collar_s=0.25).der == 0
    uem = np.zeros(1000, dtype=bool)
    uem[:300] = True
    assert diarization_error(act(REF), act(hyp), RATE, uem=uem).der == 0


def test_boundary_error_and_leakage():
    # Hypothesis turn change is 300 ms late: B's first 0.3 s leaks into A's channel.
    hyp = [Segment(0, 4.3, "A"), Segment(4.3, 8, "B")]
    s = duplex_counts(act(REF), act(hyp), RATE).summary()
    # boundaries: A on 0, A off 4 (err .3), B on 4 (err .3), B off 8
    assert s["num_boundaries"] == 4
    assert s["boundary_p90_ms"] == pytest.approx(300.0, abs=1)
    assert s["leakage"] == pytest.approx(0.3 / 8.0, abs=1e-3)


def test_overlap_scores():
    ref = [Segment(0, 5, "A"), Segment(4, 8, "B")]  # 1 s overlap
    hyp = [Segment(0, 4.5, "A"), Segment(4, 8, "B")]  # 0.5 s overlap found
    s = duplex_counts(act(ref), act(hyp), RATE).summary()
    assert s["overlap_precision"] == pytest.approx(1.0)
    assert s["overlap_recall"] == pytest.approx(0.5)


def test_bridge_gaps():
    row = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
    out = bridge_gaps(row, 2)
    assert out.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]


def test_evaluate_end_to_end(tmp_path):
    write_rttm(tmp_path / "ref.rttm", {"ep": REF})
    write_rttm(tmp_path / "hyp.rttm", {"ep": [Segment(0, 4, "x"), Segment(4, 8, "y")]})
    report = evaluate(tmp_path / "ref.rttm", tmp_path / "hyp.rttm")
    assert report["total"]["der"] == pytest.approx(0.0)
    assert report["settings"]["missing_hypotheses"] == []
