"""Real-recording test pack: two-person podcasts with automatic speaker labels, no ground truth.

Each recording has 0.5 s chunk labels (speaker 0, speaker 1, or -1 for unlabeled: pauses, turn
edges, overlaps, unsure) from ECAPA clustering, and a 30 s voice sample per speaker taken from
its longest confident stretches. Each speaker in turn is the target:

* target chunks (labeled target, outside its sample): the input is the target alone, so the
  output should equal it. SI-SDR and level of the output against the input; share of chunks
  kept (level >= -6 dB).
* other chunks (labeled the other speaker): the output should be silent. Leakage = output
  energy over input energy, pooled and per-chunk median; share of chunks silenced (<= -20 dB).

Inference as in K-S: 4 s windows, 1 s overlap, linear cross-fade, no rescaling. The pack lives
in a private Hugging Face dataset (the recordings are YouTube podcasts: internal testing only).

    python -m kezek.realtest --checkpoint kezek_tse.pt [--pack manifestro/kezek-realtest]
        [--max-s 600] [--out result.json]

From a training loop, with ``extract(mix [B, T], enr_wave [1, Te], d [1, D]) -> [B, T]``:

    from kezek import realtest
    pack = realtest.load_pack("manifestro/kezek-realtest", max_s=600)   # once
    m = realtest.evaluate(extract, se, pack, dev)                       # every eval; flat dict
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

from kezek.metrics import si_sdr

SR = 16000
PACK = "manifestro/kezek-realtest"
MIN_ENROLL_S = 20.0


@dataclass
class Item:
    id: str
    audio: np.ndarray
    chunk: np.ndarray
    chunk_n: int
    enrollment: dict[int, list[tuple[int, int]]]
    limit: int | None = None
    """Score only the first ``limit`` chunks."""


def load_pack(pack: str = PACK, max_s: float | None = None) -> list[Item]:
    """``pack``: a local folder or a Hugging Face dataset id. ``max_s``: score only the first
    ``max_s`` seconds of each recording (the voice samples may come from anywhere in it)."""
    if not os.path.isdir(pack):
        from huggingface_hub import snapshot_download

        pack = snapshot_download(pack, repo_type="dataset")
    man = json.load(open(os.path.join(pack, "manifest.json")))
    items = []
    for it in man["items"]:
        x, sr = sf.read(os.path.join(pack, it["audio"]), dtype="float32")
        assert sr == SR and x.ndim == 1, it["audio"]
        lab = json.load(open(os.path.join(pack, it["labels"])))
        n = int(lab["chunk_s"] * SR)
        enr = {  # a speaker with less than 20 s of confident speech is not a target
            int(k): [tuple(r) for r in v["enrollment_chunks"]]
            for k, v in lab["speakers"].items()
            if v["enrollment_s"] >= MIN_ENROLL_S
        }
        items.append(Item(it["id"], x, np.asarray(lab["chunks"]), n, enr))
    if max_s:
        for it in items:
            it.limit = int(max_s * SR) // it.chunk_n
    return items


@torch.no_grad()
def windowed(extract, x: np.ndarray, enr: torch.Tensor, d: torch.Tensor, dev: str, batch: int = 16):
    win, hop = 4 * SR, 3 * SR
    starts = list(range(0, len(x) - win, hop)) + [len(x) - win]
    fade = np.linspace(0.0, 1.0, win - hop, dtype=np.float32)
    out, wsum = np.zeros(len(x), np.float32), np.zeros(len(x), np.float32)
    for i in range(0, len(starts), batch):
        ss = starts[i : i + batch]
        seg = torch.from_numpy(np.stack([x[s : s + win] for s in ss])).to(dev)
        y = extract(seg, enr, d).float().cpu().numpy()
        for s, yy in zip(ss, y, strict=True):
            w = np.ones(win, np.float32)
            if s > 0:
                w[: len(fade)] = fade
            if s + win < len(x):
                w[-len(fade) :] = fade[::-1]
            out[s : s + win] += w * yy
            wsum[s : s + win] += w
    return out / np.maximum(wsum, 1e-6)


def _peak(x: np.ndarray, p: float = 0.9) -> np.ndarray:
    return (x * np.float32(p / max(float(np.abs(x).max()), 1e-6))).astype(np.float32)


@torch.no_grad()
def score(extract, se, item: Item, target: int, dev: str) -> dict:
    n = item.chunk_n
    limit = item.limit or len(item.chunk)
    x = _peak(item.audio)
    enr = np.concatenate([x[a * n : b * n] for a, b in item.enrollment[target]])
    e = torch.from_numpy(_peak(enr))[None].to(dev)
    d = se(e)
    xs = x[: limit * n]
    y = windowed(extract, xs, e, d, dev)
    excl = np.zeros(len(item.chunk), bool)
    for a, b in item.enrollment[target]:
        excl[a:b] = True
    ch = item.chunk[:limit]
    t_idx = np.flatnonzero((ch == target) & ~excl[:limit])
    o_idx = np.flatnonzero(ch == 1 - target)

    def cat(s, idx):
        return (
            np.concatenate([s[i * n : (i + 1) * n] for i in idx])
            if len(idx)
            else np.zeros(1, np.float32)
        )

    def ce(s, idx):
        return np.array([np.mean(s[i * n : (i + 1) * n] ** 2) + 1e-12 for i in idx])

    xt, yt, xo, yo = cat(xs, t_idx), cat(y, t_idx), cat(xs, o_idx), cat(y, o_idx)
    lt = 10 * np.log10(ce(y, t_idx) / ce(xs, t_idx)) if len(t_idx) else np.zeros(0)
    lo = 10 * np.log10(ce(y, o_idx) / ce(xs, o_idx)) if len(o_idx) else np.zeros(0)
    nan = float("nan")
    return {
        "target_s": len(t_idx) * n / SR,
        "other_s": len(o_idx) * n / SR,
        "target_sisdr": si_sdr(yt, xt) if len(t_idx) else nan,
        "target_level_db": float(10 * np.log10(np.mean(yt**2) / np.mean(xt**2)))
        if len(t_idx)
        else nan,
        "target_kept": float(np.mean(lt >= -6)) if len(lt) else nan,
        "leak_db": float(10 * np.log10(np.mean(yo**2) / np.mean(xo**2))) if len(o_idx) else nan,
        "leak_median_db": float(np.median(lo)) if len(lo) else nan,
        "other_silenced": float(np.mean(lo <= -20)) if len(lo) else nan,
    }


KEYS = (
    "target_sisdr",
    "target_level_db",
    "target_kept",
    "leak_db",
    "leak_median_db",
    "other_silenced",
)


def evaluate(extract, se, pack: list[Item], dev: str, per_item: bool = False) -> dict:
    """Flat dict for a tracker: ``real/<metric>`` averaged over every (recording, target) pair,
    and with ``per_item`` also ``real/<id>/s<k>/<metric>``."""
    rows = []
    for it in pack:
        for t in (0, 1):
            if it.enrollment.get(t):
                rows.append((it.id, t, score(extract, se, it, t, dev)))
    out = {f"real/{k}": float(np.nanmean([r[k] for _, _, r in rows])) for k in KEYS}
    out["real/pairs"] = len(rows)
    if per_item:
        for i, t, r in rows:
            out.update({f"real/{i}/s{t}/{k}": r[k] for k in KEYS})
    return out


def tse_extract(model):
    """Adapter for kezek.model.KezekTSE: mix [B, T], enrollment wave (unused), d [1, D]."""
    return lambda mix, enr, d: model(mix, d.expand(len(mix), -1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--pack", default=PACK)
    ap.add_argument(
        "--max-s", type=float, default=None, help="score only the first seconds of each recording"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from kezek.model import KezekTSE, SpeakerEncoder

    se = SpeakerEncoder(device=a.device)
    pack = load_pack(a.pack, a.max_s)
    res = {}
    for ck in a.checkpoint:
        m = KezekTSE().to(a.device).eval()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        r = evaluate(tse_extract(m), se, pack, a.device, per_item=True)
        res[ck] = r
        print(ck, json.dumps({k: round(r[f"real/{k}"], 2) for k in KEYS}), flush=True)
        for it in pack:
            for t in (0, 1):
                k = f"real/{it.id}/s{t}/target_sisdr"
                if k in r:
                    print(
                        f"  {it.id:14s} s{t}  "
                        + "  ".join(f"{q} {r[f'real/{it.id}/s{t}/{q}']:+.2f}" for q in KEYS)
                    )
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
