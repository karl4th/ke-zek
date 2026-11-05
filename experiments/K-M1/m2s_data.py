"""Training sequences for the two-block Kezek (M2S v0.1 signal contract).

One item = a 30 s identity-context span of one Open Yap conversation, a sequence of K adjacent
separation chunks inside it, and for each participant a clean reference clip from another
window of the same conversation (the "same speaker, other window" positive for the identity
loss and the clean-reference embedding for separator pretraining).

Signal contract: s_A = g_A a, s_B = g_B b with independent gains (+-3 dB), x = s_A + s_B, then
one common peak-safe factor c on all three; the two sources are never normalized separately and
||x - (s_A + s_B)|| is asserted to be at float precision. Sequence starts are stratified over
A-only, B-only, overlap and turn transitions.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "K-R1"))
from rdata import FRAME, _Conversation, _runs, disputed_frames  # noqa: E402

SR = 16_000
CATEGORIES = ("a_only", "b_only", "overlap", "transition")


def category_frames(act_a: np.ndarray, act_b: np.ndarray) -> dict[str, np.ndarray]:
    """Frame indices of each sampling stratum (transition = a frame where one person's run ends
    and the other's starts within 0.5 s)."""
    out = {
        "a_only": np.flatnonzero(act_a & ~act_b),
        "b_only": np.flatnonzero(act_b & ~act_a),
        "overlap": np.flatnonzero(act_a & act_b),
    }
    ends = []
    for x, y in ((act_a, act_b), (act_b, act_a)):
        for _, e in _runs(x):
            if y[e : e + 50].any():
                ends.append(e)
    out["transition"] = np.asarray(sorted(ends), np.int64)
    return out


class SpanSequences(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs,
        n: int,
        seed: int,
        span_s: float = 30.0,
        chunk_s: float = 4.0,
        chunks: int = 2,
        ref_s: float = 6.0,
        gain_db: float = 3.0,
        peak: tuple[float, float] = (0.45, 0.9),
    ):
        self.pairs, self.n, self.seed = list(pairs), n, seed
        self.span, self.chunk, self.k = int(span_s * 100), int(chunk_s * 100), chunks
        self.ref = int(ref_s * 100)
        self.gain_db, self.peak = gain_db, peak
        self._cache: dict[int, _Conversation] = {}

    def __len__(self):
        return self.n

    def _conv(self, k: int) -> _Conversation:
        if k not in self._cache:
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = _Conversation(self.pairs[k])
        return self._cache[k]

    def _reference(self, c: _Conversation, side: int, lo: int, hi: int, rng) -> np.ndarray | None:
        """Clean solo speech of one participant from outside [lo, hi), up to ref_s seconds."""
        runs = c.solo[side]
        runs = runs[(runs[:, 1] <= lo) | (runs[:, 0] >= hi)]
        if not len(runs) or (runs[:, 1] - runs[:, 0]).sum() < 200:
            return None
        pieces, got = [], 0
        for i in rng.permutation(len(runs)):
            a, b = int(runs[i, 0]), int(runs[i, 1])
            take = min(b - a, self.ref - got)
            pieces.append(c.audio(side, a, a + take))
            got += take
            if got >= self.ref:
                break
        x = np.concatenate(pieces)
        x = np.pad(x, (0, self.ref * FRAME - len(x)))[: self.ref * FRAME]
        return x * np.float32(0.5 / max(float(np.abs(x).max()), 1e-4))

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, i])
        seq = self.k * self.chunk
        for _ in range(30):
            k = int(rng.integers(len(self.pairs)))
            c = self._conv(k)
            if c.frames < self.span + 2 * self.ref + 200:
                continue
            start = int(rng.integers(0, c.frames - self.span + 1))
            sl = slice(start, start + self.span)
            act = np.stack([c.act[0][sl], c.act[1][sl]])
            cats = category_frames(act[0], act[1])
            cat = CATEGORIES[int(rng.integers(len(CATEGORIES)))]
            pool = cats[cat] if len(cats[cat]) else np.flatnonzero(act[0] | act[1])
            if not len(pool):
                continue
            centre = int(pool[int(rng.integers(len(pool)))])
            s0 = max(0, min(self.span - seq, centre - int(rng.integers(seq))))
            refs = [self._reference(c, s, start, start + self.span, rng) for s in (0, 1)]
            if refs[0] is None or refs[1] is None:
                continue
            a, b = c.audio(0, start, start + self.span), c.audio(1, start, start + self.span)
            ga, gb = (10 ** (rng.uniform(-1, 1, 2) * self.gain_db / 20)).astype(np.float32)
            sa, sb = a * ga, b * gb
            x = sa + sb
            cf = np.float32(rng.uniform(*self.peak) / max(float(np.abs(x).max()), 1e-4))
            sa, sb, x = sa * cf, sb * cf, x * cf
            assert float(np.abs(x - (sa + sb)).max()) < 1e-5, "mixture is not the sum of sources"
            levels = []
            for src, ac in ((sa, act[0]), (sb, act[1])):
                m = np.repeat(ac, FRAME)
                levels.append(float(np.mean(np.float64(src[m]) ** 2)) if m.sum() > 50 * FRAME
                              else float(np.mean(np.float64(src) ** 2)) + 1e-8)
            return {
                "mix": x.astype(np.float32),
                "src": np.stack([sa, sb]).astype(np.float32),
                "act": act,
                "disputed": disputed_frames(act[0], act[1]),
                "seq_start": np.int64(s0),
                "ref": np.stack(refs).astype(np.float32),
                "l_ref": np.asarray(levels, np.float32),
                "category": CATEGORIES.index(cat),
                "conv": self.pairs[k].conversation_id,
            }
        raise RuntimeError(f"no usable span for item {i}")


def collate(items: list[dict]) -> dict:
    out = {}
    for key in items[0]:
        if key == "conv":
            out[key] = [it[key] for it in items]
        else:
            out[key] = torch.as_tensor(np.stack([np.asarray(it[key]) for it in items]))
    return out
