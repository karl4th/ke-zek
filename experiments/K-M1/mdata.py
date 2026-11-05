"""Training sequences for the two-block Kezek: a 30 s span of one real conversation plus a few
4 s separation chunks inside it.

Both speakers are targets; their order (A/B) is arbitrary and fixed for the whole span, so the
loss chooses one permutation per span. The span gives the identity encoder its 30 s context;
the chunks are where the separator is trained. Chunks are drawn half around overlaps, half
uniformly, so overlaps, short replies and solo stretches all appear.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "K-R1"))
from rdata import FRAME, _Conversation, disputed_frames  # noqa: E402

SR = 16_000


class SpanSequences(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs,
        n: int,
        seed: int,
        span_s: float = 30.0,
        chunk_s: float = 4.0,
        chunks: int = 4,
        p_overlap: float = 0.5,
        gain_db: float = 3.0,
        peak: tuple[float, float] = (0.45, 0.9),
    ):
        self.pairs, self.n, self.seed = list(pairs), n, seed
        self.span, self.chunk, self.k = int(span_s * 100), int(chunk_s * 100), chunks
        self.p_overlap, self.gain_db, self.peak = p_overlap, gain_db, peak
        self._cache: dict[int, _Conversation] = {}

    def __len__(self):
        return self.n

    def _conv(self, k: int) -> _Conversation:
        if k not in self._cache:
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = _Conversation(self.pairs[k])
        return self._cache[k]

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, i])
        for _ in range(20):
            k = int(rng.integers(len(self.pairs)))
            c = self._conv(k)
            if c.frames < self.span + 100:
                continue
            start = int(rng.integers(0, c.frames - self.span + 1))
            sl = slice(start, start + self.span)
            act = np.stack([c.act[0][sl], c.act[1][sl]])
            if act.sum() < 300:  # under 3 s of speech in 30 s: nothing to learn from
                continue
            a, b = c.audio(0, start, start + self.span), c.audio(1, start, start + self.span)
            # one random gain per source, then one common scale for sources and mixture alike
            ga, gb = 10 ** (rng.uniform(-1, 1, 2) * self.gain_db / 20)
            a, b = a * np.float32(ga), b * np.float32(gb)
            mix = a + b
            g = np.float32(rng.uniform(*self.peak) / max(float(np.abs(mix).max()), 1e-4))
            a, b, mix = a * g, b * g, mix * g
            ov = np.flatnonzero(act[0] & act[1])
            starts = []
            for j in range(self.k):
                if j < self.k // 2 and len(ov) and rng.random() < self.p_overlap * 2:
                    s = int(ov[int(rng.integers(len(ov)))]) - int(rng.integers(self.chunk))
                else:
                    s = int(rng.integers(0, self.span - self.chunk + 1))
                starts.append(max(0, min(self.span - self.chunk, s)))
            levels = []
            for src, ac in ((a, act[0]), (b, act[1])):
                m = np.repeat(ac, FRAME)
                levels.append(float(np.mean(np.float64(src[m]) ** 2)) if m.sum() > 50 * FRAME
                              else float(np.mean(np.float64(src) ** 2)) + 1e-8)
            return {
                "mix": mix.astype(np.float32),
                "src": np.stack([a, b]).astype(np.float32),
                "act": act,
                "disputed": disputed_frames(act[0], act[1]),
                "chunks": np.asarray(starts, np.int64),
                "l_ref": np.asarray(levels, np.float32),
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
