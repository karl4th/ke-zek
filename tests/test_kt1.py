"""K-T1 code checks without torch: simulator, augmentation, reference level, metrics."""

import os

import numpy as np
import pytest

from kezek import augment, metrics
from kezek.augment import AugmentConfig, Chain, DomainBank, sample_chain
from kezek.mixing import (
    FRAME_S_CONV,
    ConversationConfig,
    active_frames,
    reference_level,
    simulate_conversation,
)

SR = 16000
FS = int(FRAME_S_CONV * SR)


def utterance(rng, seconds=10.0, level=0.1):
    """Speech-like: syllable bursts of 0.1-0.4 s with 0.05-0.3 s gaps."""
    x, t = np.zeros(int(seconds * SR)), 0
    while t < len(x):
        n = int(rng.uniform(0.1, 0.4) * SR)
        burst = rng.standard_normal(n) * np.hanning(n) * level * rng.uniform(0.5, 1.5)
        x[t : t + n] = burst[: len(x) - t]
        t += n + int(rng.uniform(0.05, 0.3) * SR)
    return x


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    sf = pytest.importorskip("soundfile")
    d = tmp_path_factory.mktemp("bank")
    rng = np.random.default_rng(0)
    for k in range(3):
        n = int(0.4 * SR)
        rir = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.06 * SR))
        rir[0] = 3.0
        os.makedirs(d / "rir", exist_ok=True)
        sf.write(d / "rir" / f"r{k}.wav", rir / np.abs(rir).max(), SR)
    for sub in ("noise", "music"):
        os.makedirs(d / "musan" / sub, exist_ok=True)
        sf.write(d / "musan" / sub / "a.wav", rng.standard_normal(5 * SR) * 0.05, SR)
    return DomainBank.from_dirs([d / "rir"], [d / "musan" / "noise"], [d / "musan" / "music"])


def speakers(seed):
    rng = np.random.default_rng(seed)
    return (
        [utterance(rng) for _ in range(2)],
        [utterance(rng) for _ in range(2)],
        utterance(rng),
        utterance(rng),
    )


NO_CODEC = AugmentConfig(codec_p=0.0)


# --- checklist 2: simulator ------------------------------------------------------------------


def test_simulator_shares_who_starts_and_sir(bank):
    cfg = ConversationConfig()
    layouts, first, bc, sir_err = [], [], [], []
    ut, uo, et, eo = speakers(1)
    for i in range(1000):
        rng = np.random.default_rng([7, i])
        e = simulate_conversation(
            ut, uo, et, eo, rng, sample_chain(bank, rng, NO_CODEC), cfg, keep_full=True
        )
        layouts.append(e.layout)
        if e.layout != 2:
            first.append(e.target_first)
            bc.append(e.backchannel)
        f = e.full
        m_t, m_o = np.repeat(f["act_t"], FS), np.repeat(f["act_o"], FS)
        if m_t.any() and m_o.any():  # a window inside one long turn has no SIR
            sir = 10 * np.log10(np.mean(f["target"][m_t] ** 2) / np.mean(f["other"][m_o] ** 2))
            sir_err.append(abs(sir - e.sir_db))
        assert len(e.mixture) == len(e.target) == 4 * SR and len(e.act_t) == 400
    share = np.bincount(layouts, minlength=3) / len(layouts)
    np.testing.assert_allclose(share, cfg.layout_p, atol=0.05)
    assert abs(np.mean(first) - 0.5) <= 0.05
    assert abs(np.mean(bc) - cfg.backchannel_p) <= 0.05
    assert max(sir_err) <= 0.1


def test_crop_starts_inside_a_turn():
    ut, uo, et, eo = speakers(2)
    for i in range(50):
        rng = np.random.default_rng([8, i])
        e = simulate_conversation(ut, uo, et, eo, rng, Chain(), keep_full=True)
        if e.layout == 2:
            continue
        first_spk, first_start, first_audio = e.full["segs"][0]
        assert first_start <= e.full["crop_start"] < first_start + len(first_audio)


# --- checklist 3: augmentation --------------------------------------------------------------


def test_one_channel_for_both_tracks(bank):
    ut, uo, et, eo = speakers(3)
    rng = np.random.default_rng(0)
    chain = Chain(clean=False, rir=bank.rir(bank.rirs[0]), phone=True, gain_db=-2.0)
    e = simulate_conversation(ut, uo, et, eo, rng, chain, keep_full=True)
    # linear chain only: the mixture is exactly the sum of the two returned tracks
    np.testing.assert_allclose(e.mixture, e.target + e.other, atol=1e-5)
    # the clean target went through the reverb and the band: it differs from the scaled dry target
    dry = e.full["dry_t"][e.full["crop_start"] : e.full["crop_start"] + 4 * SR]
    k = (dry @ e.target) / (dry @ dry)
    assert np.sum((e.target - k * dry) ** 2) > 0.01 * np.sum(e.target**2)


def test_noise_is_not_in_the_clean_target(bank):
    ut, uo, et, eo = speakers(4)
    rng = np.random.default_rng(1)
    chain = Chain(clean=False, rir=bank.rir(bank.rirs[1]), noise=bank.noises[0], snr_db=5.0)
    e = simulate_conversation(ut, uo, et, eo, rng, chain)
    resid = e.mixture - e.target - e.other
    assert np.mean(resid**2) > 0.1 * np.mean(
        (e.target + e.other) ** 2
    )  # the noise is in the mixture
    rng = np.random.default_rng(1)
    e2 = simulate_conversation(ut, uo, et, eo, rng, Chain(clean=False, rir=chain.rir))
    np.testing.assert_allclose(e.target, e2.target, rtol=1e-5, atol=1e-7)  # and not in the target


def test_clean_share_and_no_musan_speech(bank, tmp_path):
    rng = np.random.default_rng(2)
    clean = np.mean([sample_chain(bank, rng).clean for _ in range(5000)])
    assert abs(clean - 0.2) <= 0.03
    sf = pytest.importorskip("soundfile")
    os.makedirs(tmp_path / "musan" / "speech")
    sf.write(tmp_path / "musan" / "speech" / "s.wav", np.zeros(SR), SR)
    with pytest.raises(ValueError):
        DomainBank.from_dirs(noise_dirs=[tmp_path / "musan"])


@pytest.mark.skipif(not augment.has_ffmpeg(), reason="needs ffmpeg")
@pytest.mark.parametrize("kind,kbps", [("mp3", 64), ("opus", 24)])
def test_codec_keeps_alignment(kind, kbps):
    t = np.arange(3 * SR) / SR
    x = (
        0.2
        * sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
        * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    )
    y = augment.codec(x, kind, kbps)
    assert len(y) == len(x)
    scores = {lag: metrics.si_sdr(np.roll(y, lag)[100:-100], x[100:-100]) for lag in range(-8, 9)}
    assert max(scores, key=scores.get) == 0 and scores[0] > 10


# --- checklist 4: reference level -----------------------------------------------------------


def test_l_ref_falls_back_to_enrollment():
    rng = np.random.default_rng(4)
    enr = rng.standard_normal(SR) * 0.05
    l_ref = reference_level(np.zeros(SR), np.zeros(100, bool), enr, SR)
    assert l_ref > 0 and l_ref == pytest.approx(np.mean(enr**2), rel=1e-6)
    clean = utterance(rng, 3.0)
    act = active_frames(clean, SR)
    assert reference_level(clean, act, enr, SR) == pytest.approx(
        np.mean(clean[np.repeat(act, FS)] ** 2)
    )


def test_simulator_l_ref_is_from_whole_turns():
    ut, uo, et, eo = speakers(5)
    for i in range(60):
        rng = np.random.default_rng([9, i])
        e = simulate_conversation(ut, uo, et, eo, rng, Chain(), keep_full=True)
        f = e.full
        crop = f["target"][f["crop_start"] : f["crop_start"] + 4 * SR]
        if crop @ crop == 0:
            continue
        g = (crop @ e.target) / (crop @ crop)  # all gains applied after the turns were rendered
        expect = np.mean(f["target"][np.repeat(f["act_t"], FS)] ** 2) * g**2
        assert e.l_ref == pytest.approx(expect, rel=1e-4) and e.l_ref > 0
        # a crop with < 0.5 s of target: the crop alone would give another number
        if 0 < e.act_t.sum() < 50:
            crop_level = np.mean(e.target[np.repeat(e.act_t, FS)] ** 2)
            assert abs(10 * np.log10(crop_level / e.l_ref)) > 0.01


# --- checklist 7: metrics -------------------------------------------------------------------


def _window(seed=8, tail=False):
    rng = np.random.default_rng(seed)
    t, o = np.zeros(4 * SR), np.zeros(4 * SR)
    t[: int(2.6 * SR)] = utterance(rng, 2.6)
    o[int(1.8 * SR) : int(3.4 * SR)] = utterance(rng, 1.6)
    if tail:
        t[int(2.6 * SR) : int(3.0 * SR)] = (
            rng.standard_normal(int(0.4 * SR)) * 0.01 * np.exp(-np.arange(int(0.4 * SR)) / 1500)
        )
    a_t, a_o = np.zeros(400, bool), np.zeros(400, bool)
    a_t[:260], a_o[180:340] = True, True
    disp = np.zeros(400, bool)
    for e in (260, 180, 340):
        disp[e - 15 : e + 16] = True
    return t, o, a_t, a_o, disp, float(np.mean(t[: 260 * FS] ** 2))


@pytest.mark.parametrize("tail", [False, True])
def test_metrics_on_synthetic(tail):
    t, o, a_t, a_o, disp, l_ref = _window(tail=tail)
    mix = t + o
    perfect = metrics.aggregate([metrics.segment_scores(t, t, o, mix, a_t, a_o, disp, l_ref)])
    assert perfect["overlap_sisdr"] > 60 and perfect["target_solo_sisdr"] > 60
    assert perfect["other_solo_leak_db"] < -100 and perfect["both_silent_db"] < -100
    assert perfect["target_solo_level_db"] == pytest.approx(0.0, abs=1e-6)
    passthrough = metrics.aggregate([metrics.segment_scores(mix, t, o, mix, a_t, a_o, disp, l_ref)])
    assert passthrough["other_solo_leak_db"] == pytest.approx(0.0, abs=0.01)
    assert passthrough["overlap_sisdri"] == pytest.approx(0.0, abs=1e-6)
    zeros = metrics.aggregate([metrics.segment_scores(0 * t, t, o, mix, a_t, a_o, disp, l_ref)])
    assert zeros["target_solo_level_db"] < -100  # solo of the target fails
    # silence is perfect; with a reverb tail, a zero output misses the tail, R = -tail
    assert zeros["other_solo_leak_db"] < (-20 if tail else -100)
    half = metrics.aggregate([metrics.segment_scores(0.5 * t, t, o, mix, a_t, a_o, disp, l_ref)])
    assert (
        half["target_solo_level_db"] == pytest.approx(-6.02, abs=0.01)
        and half["target_solo_sisdr"] > 60
    )


def test_backchannel_recall():
    rng = np.random.default_rng(9)
    t, o = np.zeros(4 * SR), utterance(rng, 4.0)
    t[int(1.5 * SR) : int(2.1 * SR)] = utterance(rng, 0.6, level=0.2)
    a_t, a_o, disp = np.zeros(400, bool), np.ones(400, bool), np.zeros(400, bool)
    a_t[150:210] = True
    s_keep = metrics.segment_scores(t, t, o, t + o, a_t, a_o, disp, 1.0)
    s_drop = metrics.segment_scores(0.3 * t, t, o, t + o, a_t, a_o, disp, 1.0)
    assert (s_keep["bc_hits"], s_keep["bc_total"]) == (1, 1)
    assert (s_drop["bc_hits"], s_drop["bc_total"]) == (0, 1)


def test_wrong_speaker_only_on_windows_with_target():
    cos_t, cos_o = np.array([0.9, 0.1, 0.2]), np.array([0.1, 0.9, 0.9])
    share, used, skipped = metrics.wrong_speaker(cos_t, cos_o, np.array([300, 300, 100]))
    assert (share, used, skipped) == (0.5, 2, 1)
    assert metrics.wrong_speaker(cos_t, cos_t + 1, np.array([300, 300, 300]))[0] == 1.0
    assert metrics.wrong_speaker(cos_t + 1, cos_t, np.array([300, 300, 300]))[0] == 0.0
