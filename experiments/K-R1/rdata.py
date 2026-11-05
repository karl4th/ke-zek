"""Training and validation windows from real two-person conversations (Open Yap, 16 kHz arrays).

A window is a 4 s crop of one conversation. The mixture is the sum of the two clean tracks, the
target is one of them. The enrollment is a 6-10 s stretch of the *mixture* where the target talks
alone, taken from another part of the same conversation (at least ``gap_s`` away from the crop):
the same kind of voice sample a user would cut from their own recording.

Labels come from the 10 ms activity that ``convert.py`` computed on each whole clean track. The
item has the same keys as the K-T1 simulator, so the K-T1 loss and validation apply unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from kezek.open_yap import ConversationPair, check_speaker_disjoint

SR = 16_000
FRAME = 160
DISPUTED_FRAMES = 15  # 0.15 s around each activity edge, as in the K-T1 metrics


def scan(root: str | Path) -> list[ConversationPair]:
    pairs = []
    for meta_path in sorted(Path(root).glob("conv_*/meta.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        frames = int(np.load(d / "a.npy", mmap_mode="r").shape[0])
        pairs.append(
            ConversationPair(
                conversation_id=str(meta["conversation_id"]),
                speaker_a_id=str(meta["speaker_a_id"]),
                speaker_b_id=str(meta["speaker_b_id"]),
                speaker_a_path=d / "a.npy",
                speaker_b_path=d / "b.npy",
                sample_rate=SR,
                frames=frames,
            )
        )
    if not pairs:
        raise FileNotFoundError(f"no converted conversations under {root}")
    return pairs


def fixed_splits(root: str | Path, split_file: str | Path):
    """Validation and test exactly as in ``split_file``; train is every other conversation
    whose speakers appear in neither (data added later can only grow train)."""
    ids = json.loads(Path(split_file).read_text())
    pairs = {p.conversation_id: p for p in scan(root)}
    missing = [c for n in ("val", "test") for c in ids[n] if c not in pairs]
    if missing:
        raise ValueError(f"{len(missing)} validation/test conversations are missing, e.g. {missing[:3]}")
    val = [pairs[c] for c in ids["val"]]
    test = [pairs[c] for c in ids["test"]]
    held = {c for c in ids["val"] + ids["test"]}
    held_speakers = {s for p in val + test for s in p.speaker_ids}
    train = sorted(
        (p for c, p in pairs.items() if c not in held and not p.speaker_ids & held_speakers),
        key=lambda p: p.conversation_id,
    )
    check_speaker_disjoint(train, val, test)
    return train, val, test


def splits(root: str | Path, seed: int = 42, fractions=(0.8, 0.1, 0.1), split_file=None):
    if split_file:
        return fixed_splits(root, split_file)
    return _connected_splits(root, seed, fractions)


def _connected_splits(root: str | Path, seed: int = 42, fractions=(0.8, 0.1, 0.1)):
    """Speaker-disjoint train / validation / test over connected identity groups.

    Speakers recur across Open Yap conversations, so most conversations form one connected
    group. That group always goes to train; the other groups are shuffled and fill validation
    and test up to their share of conversations, the rest goes to train."""
    pairs = scan(root)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        parent[find(p.speaker_a_id)] = find(p.speaker_b_id)
    groups: dict[str, list[ConversationPair]] = {}
    for p in pairs:
        groups.setdefault(find(p.speaker_a_id), []).append(p)
    comps = sorted(groups.values(), key=lambda g: (-len(g), g[0].conversation_id))
    rest = comps[1:]
    order = np.random.default_rng(seed).permutation(len(rest))
    out: list[list[ConversationPair]] = [list(comps[0]), [], []]
    target = [0, round(fractions[1] * len(pairs)), round(fractions[2] * len(pairs))]
    for i in order:
        g = rest[i]
        need = [target[k] - len(out[k]) for k in (1, 2)]
        k = 1 + int(np.argmax(need))
        out[k if need[k - 1] >= len(g) / 2 else 0].extend(g)
    for s in out:
        s.sort(key=lambda p: p.conversation_id)
    check_speaker_disjoint(*out)
    return out[0], out[1], out[2]


def split_ids(split) -> dict[str, list[str]]:
    return {n: [p.conversation_id for p in s] for n, s in zip(("train", "val", "test"), split)}


def _runs(mask: np.ndarray) -> np.ndarray:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2)


def disputed_frames(act_t: np.ndarray, act_o: np.ndarray) -> np.ndarray:
    """Frames within 0.15 s of an activity edge of either speaker (gaps < 0.3 s closed first)."""
    disputed = np.zeros(len(act_t), bool)
    for a in (act_t, act_o):
        b = a.copy()
        idx = np.flatnonzero(b)
        for i, j in zip(idx[:-1], idx[1:], strict=False):
            if 1 < j - i <= 30:
                b[i:j] = True
        for e in np.flatnonzero(np.diff(b.astype(np.int8))) + 1:
            disputed[max(0, e - DISPUTED_FRAMES) : e + DISPUTED_FRAMES + 1] = True
    return disputed


class _Conversation:
    """Memory-mapped tracks and cached activity of one conversation."""

    def __init__(self, pair: ConversationPair):
        d = pair.speaker_a_path.parent
        self.tracks = (np.load(d / "a.npy", mmap_mode="r"), np.load(d / "b.npy", mmap_mode="r"))
        self.act = (np.load(d / "act_a.npy").astype(bool), np.load(d / "act_b.npy").astype(bool))
        n = min(len(self.act[0]), len(self.act[1]), len(self.tracks[0]) // FRAME)
        self.act = (self.act[0][:n], self.act[1][:n])
        self.frames = n
        self.overlap = np.flatnonzero(self.act[0] & self.act[1])
        # target-only runs of at least 0.5 s, per side: candidate enrollment pieces
        self.solo = []
        for s in (0, 1):
            r = _runs(self.act[s] & ~self.act[1 - s])
            self.solo.append(r[(r[:, 1] - r[:, 0]) >= 50])

    def audio(self, side: int, a: int, b: int) -> np.ndarray:
        return np.asarray(self.tracks[side][a * FRAME : b * FRAME], dtype=np.float32) / 32768.0


class RealWindows(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: list[ConversationPair],
        n: int,
        seed: int,
        crop_s: float = 4.0,
        enroll_s: tuple[float, float] = (6.0, 10.0),
        gap_s: float = 2.0,
        p_overlap: float = 0.5,
        sir_jitter_db: float = 0.0,
        peak: tuple[float, float] = (0.45, 0.9),
    ):
        self.pairs, self.n, self.seed = list(pairs), n, seed
        self.crop = int(round(crop_s * 100))
        self.enroll_s, self.gap = enroll_s, int(round(gap_s * 100))
        self.p_overlap, self.sir_jitter_db, self.peak = p_overlap, sir_jitter_db, peak
        self._cache: dict[int, _Conversation] = {}

    def __len__(self):
        return self.n

    def _conv(self, k: int) -> _Conversation:
        if k not in self._cache:
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = _Conversation(self.pairs[k])
        return self._cache[k]

    def _enrollment(self, c: _Conversation, side: int, start: int, rng) -> tuple[np.ndarray, int]:
        """Mixture audio of shuffled target-solo runs outside the crop and its guard band."""
        need = int(round(rng.uniform(*self.enroll_s) * 100))
        runs = c.solo[side]
        lo, hi = start - self.gap, start + self.crop + self.gap
        runs = runs[(runs[:, 1] <= lo) | (runs[:, 0] >= hi)]
        if len(runs) == 0 or (runs[:, 1] - runs[:, 0]).sum() < min(need, 300):
            raise ValueError("not enough solo speech for an enrollment")
        pieces, got = [], 0
        for i in rng.permutation(len(runs)):
            a, b = int(runs[i, 0]), int(runs[i, 1])
            take = min(b - a, need - got)
            off = int(rng.integers(0, b - a - take + 1))
            pieces.append((a + off, a + off + take))
            got += take
            if got >= need:
                break
        pieces.sort()
        enr = np.concatenate([c.audio(0, a, b) + c.audio(1, a, b) for a, b in pieces])
        return enr, got

    @staticmethod
    def _level(c: _Conversation, side: int, x: np.ndarray, act: np.ndarray, g: float, rng) -> float:
        """L_ref: energy of the clean track on its active frames in the window; if it is nearly
        silent here, on a stretch of its own solo speech elsewhere (same gain)."""
        m = np.repeat(act, FRAME)
        if m.sum() >= 50 * FRAME:
            return float(np.mean(np.float64(x[m]) ** 2))
        r = c.solo[side]
        if not len(r):
            return float(np.mean(np.float64(x) ** 2)) + 1e-8
        a, b = (int(v) for v in r[int(rng.integers(len(r)))])
        return float(np.mean(np.float64(c.audio(side, a, b) * g) ** 2)) + 1e-12

    def item(self, i: int, tries: int = 20) -> dict:
        rng = np.random.default_rng([self.seed, i])
        for _ in range(tries):
            k = int(rng.integers(len(self.pairs)))
            c = self._conv(k)
            if c.frames < self.crop + 2 * self.gap + 600:
                continue
            side = int(rng.integers(2))
            if rng.random() < self.p_overlap and len(c.overlap):
                f = int(c.overlap[int(rng.integers(len(c.overlap)))])
                start = f - int(rng.integers(self.crop))
            else:
                start = int(rng.integers(c.frames - self.crop + 1))
            start = max(0, min(c.frames - self.crop, start))
            try:
                enr, _ = self._enrollment(c, side, start, rng)
            except ValueError:
                continue
            fr = slice(start, start + self.crop)
            t = c.audio(side, start, start + self.crop)
            o = c.audio(1 - side, start, start + self.crop)
            act_t, act_o = c.act[side][fr].copy(), c.act[1 - side][fr].copy()
            if self.sir_jitter_db:
                o = o * np.float32(10 ** (rng.uniform(-1, 1) * self.sir_jitter_db / 20))
            mix = t + o
            g = rng.uniform(*self.peak) / max(float(np.abs(mix).max()), 1e-4)
            t, o, mix = t * g, o * g, mix * g
            l_ref = self._level(c, side, t, act_t, g, rng)
            l_ref_o = self._level(c, 1 - side, o, act_o, g, rng)
            enr = enr * np.float32(0.9 / max(float(np.abs(enr).max()), 1e-4))
            # the other speaker's enrollment, for the wrong-speaker check in validation
            try:
                enr_o, _ = self._enrollment(c, 1 - side, start, rng)
                enr_o = enr_o * np.float32(0.9 / max(float(np.abs(enr_o).max()), 1e-4))
            except ValueError:
                enr_o = enr
            disputed = disputed_frames(act_t, act_o)
            e_t = float(np.mean(np.float64(t) ** 2)) + 1e-12
            e_o = float(np.mean(np.float64(o) ** 2)) + 1e-12
            return {
                "mix": mix.astype(np.float32),
                "clean": t.astype(np.float32),
                "other": o.astype(np.float32),
                "act_t": act_t,
                "act_o": act_o,
                "disputed": disputed,
                "silent": ~act_t & ~disputed,
                "l_ref": np.float32(l_ref),
                "l_ref_o": np.float32(l_ref_o),
                "silent_o": ~act_o & ~disputed,
                "enr_t": enr.astype(np.float32),
                "enr_o": enr_o.astype(np.float32),
                "clean_window": True,
                "sir_db": np.float32(10 * np.log10(e_t / e_o)),
                "layout": "real",
                "conv": self.pairs[k].conversation_id,
                "side": side,
                "start": start,
            }
        raise RuntimeError(f"no usable window for item {i} after {tries} tries")

    def __getitem__(self, i):
        return self.item(i)


def collate(items: list[dict]) -> dict:
    """Enrollments differ in length: cut to the shortest in the batch."""
    out = {}
    for k in items[0]:
        if k in ("enr_t", "enr_o"):
            n = min(len(it[k]) for it in items)
            out[k] = torch.from_numpy(np.stack([it[k][:n] for it in items]))
        elif k in ("layout", "conv"):
            out[k] = [it[k] for it in items]
        else:
            out[k] = torch.as_tensor(np.stack([np.asarray(it[k]) for it in items]))
    return out
