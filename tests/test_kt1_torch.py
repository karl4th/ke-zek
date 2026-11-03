"""K-T1 checks that need torch: loss scale and silence, watchdog, test lists only with --final."""

import importlib.util
import inspect
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from kezek.losses import kt1_loss, residual_db, snr_tau  # noqa: E402

SR = 16000
KT1 = os.path.join(os.path.dirname(__file__), "..", "experiments", "K-T1")


def _batch(rng, reverb_tail=False):
    s = np.zeros(SR)
    s[: SR // 2] = rng.standard_normal(SR // 2) * 0.1
    act = np.zeros(100, bool)
    act[:50] = True
    if reverb_tail:  # the clean target rings on after the dry turn ends
        s[SR // 2 : SR // 2 + 4000] = (
            rng.standard_normal(4000) * 0.03 * np.exp(-np.arange(4000) / 1000)
        )
    silent = ~act
    silent[35:66] = False  # disputed frames around the boundary
    l_ref = np.mean(s[: SR // 2] ** 2)

    def t(x):
        return torch.as_tensor(np.asarray(x)[None])

    return t(np.float32(s)), t(act), t(silent), t(np.float32([l_ref]))[0]


# --- checklist 5: the active-frame term fixes the output scale --------------------------------


@pytest.mark.parametrize("k,expect", [(1.0, 30.0), (0.5, 6.0), (1.5, 6.0), (2.0, 0.0)])
def test_snr_term_fixes_the_scale(k, expect):
    s, act, _, _ = _batch(np.random.default_rng(5))
    w = act.float().repeat_interleave(160, -1)
    assert float(snr_tau(k * s, s, w)) == pytest.approx(expect, abs=0.1)


def test_loss_module_has_no_si_sdr():
    import kezek.losses as L

    src = inspect.getsource(L)
    assert "si_sdr" not in src and "metrics" not in src


# --- checklist 6: silence ----------------------------------------------------------------------


def test_silence_penalty():
    s, act, silent, l_ref = _batch(np.random.default_rng(6))
    other = torch.zeros_like(s)
    other[0, SR // 2 :] = torch.randn(SR // 2) * 0.1
    w = silent.float().repeat_interleave(160, -1)
    # output = input: penalised
    assert float(residual_db(s + other, s, w, l_ref)) + 30 > 0
    # output = 0: no penalty
    assert float(torch.relu(residual_db(torch.zeros_like(s), s, w, l_ref) + 30)) == 0.0
    # output = clean target with its reverb tail: no penalty
    s_t, _, silent_t, l_t = _batch(np.random.default_rng(6), reverb_tail=True)
    w_t = silent_t.float().repeat_interleave(160, -1)
    assert float(torch.relu(residual_db(s_t, s_t, w_t, l_t) + 30)) == 0.0


def test_loss_ignores_disputed_frames_and_silence_in_snr():
    s, act, silent, l_ref = _batch(np.random.default_rng(7))
    est = s.clone()
    est[0, 8000:10400] += 1.0  # junk only in silent frames 50-64, all of them disputed
    loss, parts = kt1_loss(est, s, act, silent, l_ref, weights=(1.0, 0.3, 0.0))
    assert parts["silence"] == 0.0 and parts["snr_active"] == pytest.approx(30.0, abs=0.1)


# --- checklist 11-12: watchdog, test lists only with --final -----------------------------------


def _load(name):
    sys.path.insert(0, KT1)
    spec = importlib.util.spec_from_file_location(name, os.path.join(KT1, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_watchdog():
    ev = _load("evaluate")
    assert not ev.watchdog_tripped([(5000, 4.0)])  # 5k is observation only
    assert not ev.watchdog_tripped([(5000, 4.0), (10000, 4.0)])
    assert ev.watchdog_tripped([(5000, 4.0), (10000, 4.0), (15000, -4.0)])
    assert not ev.watchdog_tripped([(10000, 4.0), (15000, 2.0)])


def test_paired_delta_is_tighter_than_two_intervals():
    ev = _load("evaluate")
    rng = np.random.default_rng(11)
    base = rng.normal(12.8, 8.0, 300)  # mixtures differ a lot in difficulty
    new = base - 0.3 + rng.normal(0, 0.5, 300)  # the model change is small and consistent
    d = ev.paired_delta(new, base)
    lo, hi = d["delta_ci"]
    assert lo < -0.3 < hi and hi - lo < 0.2  # an unpaired comparison would be ~±0.9 dB wide


def test_test_lists_need_final(tmp_path):
    ev = _load("evaluate")
    lst = str(tmp_path / "t.json")
    with pytest.raises(SystemExit):
        ev.parse(["--checkpoint", "x.pt", "--val-root", "v", "--tests", lst])
    with pytest.raises(SystemExit):
        ev.parse(["--checkpoint", "x.pt", "--val-root", "v", "--dev-clean-300", "dc"])
    a = ev.parse(["--checkpoint", "x.pt", "--final", "--tests", lst])
    assert a.final and a.tests
