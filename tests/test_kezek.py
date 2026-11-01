import numpy as np
import pytest

from kezek.cli import main as cli_main
from kezek.enrollment import pick_enrollment
from kezek.metrics import regions, score_extraction, si_sdr
from kezek.mixing import conversation_schedule, mix_pair, render, rms_db, set_rms_db
from kezek.vad import energy_vad

SR = 16000


def burst(rng, on_s, off_s, total_s, freq=None):
    """Noise (or tone) active on [on_s, off_s), silent elsewhere."""
    x = np.zeros(int(total_s * SR))
    a, b = int(on_s * SR), int(off_s * SR)
    if freq is None:
        x[a:b] = rng.standard_normal(b - a) * 0.1
    else:
        x[a:b] = 0.1 * np.sin(2 * np.pi * freq * np.arange(b - a) / SR)
    return x


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    # target talks 0-4 s, interferer 3-7 s: overlap 3-4, solo 0-3, absent 4-7
    return burst(rng, 0, 4, 8), burst(rng, 3, 7, 8)


def test_si_sdr_basics():
    rng = np.random.default_rng(1)
    s = rng.standard_normal(SR)
    assert si_sdr(s, s) > 90
    assert si_sdr(3.0 * s, s) > 90  # scale invariant
    n = rng.standard_normal(SR)
    assert si_sdr(s + n, s) == pytest.approx(0.0, abs=0.3)
    with pytest.raises(ValueError):
        si_sdr(s, s[:-1])


def test_regions(pair):
    t, i = pair
    r = regions(t, i, SR)
    sec = r.seconds()
    assert sec["overlap_s"] == pytest.approx(1.0, abs=0.25)
    assert sec["solo_s"] == pytest.approx(3.0, abs=0.25)
    assert sec["absent_s"] == pytest.approx(3.0, abs=0.25)


def test_perfect_mixture_and_silence(pair):
    t, i = pair
    perfect = score_extraction(t, t, i, SR)
    assert perfect.sisdr_overlap > 60
    assert perfect.sisdr_solo > 60
    assert perfect.leakage == 0.0
    assert perfect.residual_db < -60

    unprocessed = score_extraction(t + i, t, i, SR)
    assert unprocessed.sisdri_overlap == pytest.approx(0.0, abs=1e-6)
    assert unprocessed.leakage == pytest.approx(1.0, abs=0.05)
    assert unprocessed.residual_db == pytest.approx(0.0, abs=0.1)


def test_vad_hangover():
    rng = np.random.default_rng(2)
    x = burst(rng, 1, 2, 3)
    assert energy_vad(x, SR, hangover_s=0.0).sum() == 50  # 1 s of 20 ms frames
    assert energy_vad(x, SR, hangover_s=0.1).sum() == 60


def test_mix_pair_sir_and_offset():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(SR), rng.standard_normal(SR // 2) * 5
    mix, t, i = mix_pair(a, b, sir_db=5.0, offset=SR // 4)
    assert len(mix) == len(t) == len(i) == SR
    np.testing.assert_allclose(mix, t + i)
    assert rms_db(t[t != 0]) - rms_db(i[i != 0]) == pytest.approx(5.0, abs=1e-6)
    mix, t, i = mix_pair(a, b, offset=-SR // 4)
    assert len(mix) == SR + SR // 4
    assert set_rms_db(a, -20.0) is not a and rms_db(set_rms_db(a, -20.0)) == pytest.approx(-20)


def test_conversation_schedule_alternates_and_overlaps_are_bounded():
    rng = np.random.default_rng(4)
    da, db = [2.0, 3.0, 1.0], [1.5, 0.5]
    sched = conversation_schedule(da, db, rng, overlap_prob=0.5)
    assert [p.speaker for p in sched] == [0, 1, 0, 1, 0]
    clips_a = [np.ones(int(d * SR)) for d in da]
    clips_b = [np.ones(int(d * SR)) for d in db]
    mix, t, i = render(sched, clips_a, clips_b, SR)
    np.testing.assert_allclose(mix, t + i)
    assert t.sum() == pytest.approx(sum(da) * SR, rel=1e-3)
    overlap_s = ((t > 0) & (i > 0)).sum() / SR
    assert overlap_s <= 0.5 * sum(min(x, y) for x, y in zip(da, db, strict=False)) + 1e-6


def test_pick_enrollment_excludes_test_segment(pair):
    t, i = pair
    r = regions(t, i, SR)
    audio, used = pick_enrollment(t + i, r.solo, SR, duration_s=1.5, exclude_s=(2.0, 8.0))
    assert len(audio) == int(1.5 * SR)
    assert all(e <= 2.0 for _, e in used)
    with pytest.raises(ValueError):
        pick_enrollment(t + i, r.solo, SR, duration_s=5.0)


def test_cli_prepare_and_score(tmp_path):
    sf = pytest.importorskip("soundfile")
    rng = np.random.default_rng(5)
    a = burst(rng, 0, 5, 12) + burst(rng, 8, 12, 12)
    b = burst(rng, 4, 9, 12)
    sf.write(tmp_path / "conv.wav", np.stack([a, b], axis=1), SR, subtype="FLOAT")
    item = tmp_path / "item"
    assert cli_main(["prepare", str(tmp_path / "conv.wav"), "--target-channel", "0",
                     "--out", str(item), "--start", "6", "--enroll", "2"]) == 0  # fmt: skip
    for name in ["mixture", "target", "interferer", "enrollment"]:
        assert (item / f"{name}.wav").exists()
    target, _ = sf.read(item / "target.wav")
    sf.write(item / "estimate.wav", target, SR, subtype="FLOAT")
    assert cli_main(["score", str(item), "--json", str(tmp_path / "r.json")]) == 0
