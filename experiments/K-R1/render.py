"""Make a summed Open Yap pair sound like one mono recording of a conversation.

Open Yap gives each person a clean close-microphone track; the recordings Kezek has to split are
one mono signal from a room, a mixing desk and a lossy pipeline. ``Render`` takes the two clean
tracks of a window and applies one recording chain to both:

* per person (linear, part of that person's reference): room reverb (one RIR for both: same
  room), recording EQ (low and high shelves: microphone, desk, mastering), microphone bleed (a delayed, attenuated copy of the voice, as picked up by the other
  person's microphone), telephone band;
* on the sum, applied to both references too so that mixture = sum of references still holds:
  dynamic-range compression (the gain curve computed from the mixture's envelope);
* on the mixture only (never in a reference): background noise and music, gain, clipping, lossy
  codec (K-T1 ``Chain.finish``).

The enrollment is cut from the same recording, so it goes through the same chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kezek.augment import AugmentConfig, Chain, DomainBank, sample_chain

SR = 16_000


@dataclass(frozen=True)
class RenderConfig:
    eq_p: float = 0.7
    low_shelf_db: tuple[float, float] = (-6.0, 12.0)
    low_shelf_hz: tuple[float, float] = (100.0, 300.0)
    high_shelf_db: tuple[float, float] = (-12.0, 3.0)
    high_shelf_hz: tuple[float, float] = (2500.0, 6000.0)
    bleed_p: float = 0.5
    bleed_delay_ms: tuple[float, float] = (1.0, 15.0)
    bleed_db: tuple[float, float] = (-25.0, -8.0)
    compress_p: float = 0.7
    compress_ratio: tuple[float, float] = (2.0, 10.0)
    compress_thresh_db: tuple[float, float] = (-30.0, -15.0)


@dataclass
class Recording:
    chain: Chain
    bleed: tuple[int, float] | None = None  # delay in samples, linear gain
    compress: tuple[float, float] | None = None  # threshold dB below peak, ratio
    eq: tuple[float, float, float, float] | None = None  # low dB, low Hz, high dB, high Hz

    def sources(self, x: np.ndarray) -> np.ndarray:
        y = self.chain.sources(x)
        if self.eq is not None:
            lo_db, lo_hz, hi_db, hi_hz = self.eq
            y = shelf(y, lo_hz, lo_db, "low")
            y = shelf(y, hi_hz, hi_db, "high")
        if self.bleed is not None:
            d, a = self.bleed
            y = y + a * np.concatenate([np.zeros(d), y[:-d]]) if d < len(y) else y
        return y

    def compression_gain(self, mix: np.ndarray) -> np.ndarray:
        """Per-sample gain of a feed-forward compressor driven by the mixture (10 ms RMS,
        5 ms attack / 80 ms release on the gain in dB)."""
        if self.compress is None:
            return np.ones(len(mix))
        thr_rel, ratio = self.compress
        n = 160
        f = len(mix) // n + 1
        env = np.sqrt(np.convolve(np.pad(mix, (0, f * n - len(mix))) ** 2, np.ones(n) / n, mode="same")[::n] + 1e-12)
        level = 20 * np.log10(env)
        thr = level.max() + thr_rel
        target = np.where(level > thr, (thr - level) * (1 - 1 / ratio), 0.0)
        g = np.zeros_like(target)
        for i in range(1, len(target)):  # attack fast, release slow
            k = 0.86 if target[i] < g[i - 1] else 0.12
            g[i] = g[i - 1] + k * (target[i] - g[i - 1])
        gain = 10 ** (g / 20)
        return np.interp(np.arange(len(mix)), np.arange(len(gain)) * n, gain)


def shelf(x: np.ndarray, f0: float, gain_db: float, kind: str) -> np.ndarray:
    """RBJ shelving biquad (S = 1): the colouring of a microphone, desk or mastering EQ."""
    from scipy.signal import lfilter

    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    cw, alpha = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2)
    sa = 2 * np.sqrt(A) * alpha
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    else:
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return lfilter(np.asarray(b) / a[0], np.asarray(a) / a[0], x)


class Render:
    def __init__(self, bank: DomainBank, cfg: AugmentConfig | None = None, rcfg: RenderConfig | None = None):
        self.bank, self.cfg, self.rcfg = bank, cfg or AugmentConfig(), rcfg or RenderConfig()

    def sample(self, rng: np.random.Generator) -> Recording:
        chain = sample_chain(self.bank, rng, self.cfg)
        rec = Recording(chain)
        if chain.clean:
            return rec
        r = self.rcfg
        if rng.random() < r.eq_p:
            rec.eq = (float(rng.uniform(*r.low_shelf_db)), float(rng.uniform(*r.low_shelf_hz)),
                      float(rng.uniform(*r.high_shelf_db)), float(rng.uniform(*r.high_shelf_hz)))
        if rng.random() < r.bleed_p:
            rec.bleed = (int(rng.uniform(*r.bleed_delay_ms) * SR / 1000), float(10 ** (rng.uniform(*r.bleed_db) / 20)))
        if rng.random() < r.compress_p:
            rec.compress = (float(rng.uniform(*r.compress_thresh_db)), float(rng.uniform(*r.compress_ratio)))
        return rec

    def apply(self, rec: Recording, a: np.ndarray, b: np.ndarray, speech_mask: np.ndarray, rng):
        """Clean tracks a, b of one window -> (ref_a, ref_b, mixture), float32, same length;
        mixture = ref_a + ref_b plus whatever only the recording adds (noise, codec, clipping)."""
        if rec.chain.clean:
            return a, b, a + b
        ra, rb = rec.sources(a)[: len(a)], rec.sources(b)[: len(b)]
        g = rec.compression_gain(ra + rb)
        ra, rb = ra * g, rb * g
        m = ra + rb
        e = float(np.mean(m[speech_mask] ** 2)) if speech_mask.any() else float(np.mean(m**2)) + 1e-12
        mix, gain = rec.chain.finish(m, e, rng)
        mix = mix[: len(a)]
        if len(mix) < len(a):
            mix = np.pad(mix, (0, len(a) - len(mix)))
        return (ra * gain).astype(np.float32), (rb * gain).astype(np.float32), mix.astype(np.float32)
