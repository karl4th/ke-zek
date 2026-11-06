"""Long-form inference of the two-block Kezek (M2S v0.1 section 5): mono -> A, B, stereo.

    python experiments/K-M1/m2s_infer.py --model model.pt --input mono.wav --out-dir out/

1. Identity pass: 30 s windows with a 15 s hop give slot activity and two slot embeddings each.
2. Linking (``link_windows``): the slot order of every window is decided jointly by Viterbi
   from overlap continuity between neighbours and slot embeddings against two recording
   centroids. The older greedy slot memory (``build_memory``) is kept for comparison: it drifts
   and swaps A/B on long recordings.
3. Separation: 8 s chunks with 2 s overlap, conditioned on the final memory embeddings (fixed for
   the whole recording, so the A/B mapping cannot change inside it), cross-faded.
Outputs keep the input's sample count; the manifest lists windows, their assignment, margin,
confidence and uncertain spans.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m2s_model import ConditionedSeparator, IdentityEncoder  # noqa: E402

SR = 16_000


class TwoBlock(torch.nn.Module):
    def __init__(self, speaker=None, features="mel", backbone="lstm", act_head="multilabel"):
        super().__init__()
        self.enc = IdentityEncoder(speaker=speaker, features=features, backbone=backbone, act_head=act_head)
        self.sep = ConditionedSeparator(emb=self.enc.dim)


def load(path: str, device: str) -> TwoBlock:
    state = torch.load(path, map_location="cpu")
    speaker = None
    if state.get("slot_embedding", "learned") == "ecapa":
        from kezek.model import SpeakerEncoder

        speaker = SpeakerEncoder(device=device)
    m = TwoBlock(speaker, state.get("features", "mel"), state.get("backbone", "lstm"),
                 state.get("act_head", "multilabel")).to(device).eval()
    m.enc.solo_threshold = float(state.get("solo_threshold", 0.0))
    m.enc.load_state_dict(state["enc"])
    m.sep.load_state_dict(state["sep"])
    return m


@torch.no_grad()
def identity_pass(model, x: np.ndarray, device: str, win_s=30.0, hop_s=15.0):
    win, hop = int(win_s * SR), int(hop_s * SR)
    n = len(x)
    starts = list(range(0, max(n - win, 0) + 1, hop))
    if starts[-1] + win < n:
        starts.append(max(n - win, 0))
    out = []
    for s in starts:
        seg = torch.from_numpy(np.ascontiguousarray(x[s:s + win]))[None].to(device)
        logits, z = model.enc(seg)
        p = torch.sigmoid(logits[0])
        solo = torch.stack([p[0] * (1 - p[1]), p[1] * (1 - p[0])])
        out.append({"start": s, "end": min(s + win, n), "z": z[0].cpu(),
                    "solo_s": (solo.sum(-1) * 0.02).cpu().tolist(), "p": p.cpu()})
    return out


def build_memory(windows, distinct=0.5, margin=0.2, ema=0.1, min_solo=2.0, upd_solo=1.0):
    mem = [None, None]
    for w in windows:
        z = F.normalize(w["z"], dim=-1)
        solo = w["solo_s"]
        w["assign"], w["margin"], w["confident"] = None, 0.0, False
        if mem[0] is None:
            if min(solo) >= min_solo and float(z[0] @ z[1]) < distinct:
                mem = [z[0].clone(), z[1].clone()]
                w["assign"], w["confident"], w["margin"] = [0, 1], True, 1.0
            elif max(solo) >= min_solo:
                i = int(np.argmax(solo))
                mem[0] = z[i].clone()
                w["assign"], w["confident"] = [0, 1] if i == 0 else [1, 0], True
            continue
        if mem[1] is None:
            j = int(np.argmin([float(z[i] @ mem[0]) for i in (0, 1)]))
            if solo[j] >= min_solo and float(z[j] @ mem[0]) < distinct:
                mem[1] = z[j].clone()
            w["assign"] = [1 - j, j]  # slot of A, slot of B
            w["confident"] = mem[1] is not None
            continue
        keep = float(z[0] @ mem[0] + z[1] @ mem[1])
        swap = float(z[1] @ mem[0] + z[0] @ mem[1])
        w["margin"] = abs(keep - swap) / 2
        a_slot = 0 if keep >= swap else 1
        w["assign"] = [a_slot, 1 - a_slot]
        w["confident"] = w["margin"] >= margin
        if w["confident"]:
            for m, s in enumerate(w["assign"]):
                if solo[s] >= upd_solo:
                    mem[m] = F.normalize((1 - ema) * mem[m] + ema * z[s], dim=-1)
    if mem[1] is None:
        mem[1] = -mem[0] if mem[0] is not None else None  # one voice only: B stays silent
    return mem


def _solo(w):
    p = w["p"].float()
    return torch.stack([p[0] * (1 - p[1]), p[1] * (1 - p[0])])


def link_windows(windows, overlap_w=1.0, emb_w=4.0, solo_cap=10.0, iters=4, margin=1.0):
    """Global A/B linking of all windows at once (instead of the greedy memory):
    a binary chain over windows (keep or swap their slot order) solved exactly by Viterbi.
    * Neighbours agree through their overlap: seconds of solo activity that line up when both
      keep the same order versus the crossed order (``overlap_w`` per second).
    * Each window agrees with two recording centroids through its slot embeddings, weighted by
      the slot's solo seconds (capped at ``solo_cap``), ``emb_w`` per unit of cosine.
    Centroids start from the window with the most solo speech of both voices (or continuity
    alone) and are re-estimated from the decoded order.
    Sets w["assign"], w["margin"] (what the best decoding loses by flipping the cheapest run of
    windows that contains this one)
    and w["confident"] (margin >= ``margin``) like build_memory and returns the centroids."""
    n = len(windows)
    if n == 0:
        return [None, None]
    z = torch.stack([F.normalize(w["z"].float(), dim=-1) for w in windows])  # [n, 2, D]
    solo = [_solo(w) for w in windows]
    wt = torch.tensor([[min(s, solo_cap) / solo_cap if s >= 0.5 else 0.0 for s in w["solo_s"]] for w in windows])
    pair = np.zeros(n - 1, np.float64)  # (keep - cross) seconds of agreement with the next window
    for t in range(n - 1):
        off = (windows[t + 1]["start"] - windows[t]["start"]) // 320
        a, b = solo[t][:, off:], solo[t + 1]
        L = min(a.shape[1], b.shape[1])
        if L <= 0:
            continue
        a, b = a[:, :L], b[:, :L]
        keep = float((a[0] * b[0]).sum() + (a[1] * b[1]).sum())
        cross = float((a[0] * b[1]).sum() + (a[1] * b[0]).sum())
        pair[t] = overlap_w * (keep - cross) * 0.02
    unary = np.zeros((n, 2))
    t0 = int(wt.min(1).values.argmax())  # start from the window where both voices talk alone most
    cent = z[t0].clone() if float(wt[t0].min()) > 0 else None
    for it in range(iters + 1):
        if cent is not None:
            sim = torch.einsum("nsd,kd->nsk", z, cent)  # slot s vs centroid k
            unary[:, 0] = emb_w * (wt[:, 0] * sim[:, 0, 0] + wt[:, 1] * sim[:, 1, 1]).numpy()
            unary[:, 1] = emb_w * (wt[:, 0] * sim[:, 0, 1] + wt[:, 1] * sim[:, 1, 0]).numpy()
        # Viterbi; state 0 keeps the window's order (slot 0 -> A), 1 swaps it
        score = unary[0].copy()
        back = np.zeros((n, 2), np.int64)
        for t in range(1, n):
            new = np.empty(2)
            for s in (0, 1):
                cand = [score[q] + (pair[t - 1] if q == s else -pair[t - 1]) for q in (0, 1)]
                back[t, s] = int(np.argmax(cand))
                new[s] = max(cand) + unary[t, s]
            score = new
        st = np.zeros(n, np.int64)
        st[-1] = int(np.argmax(score))
        for t in range(n - 1, 0, -1):
            st[t - 1] = back[t, st[t]]
        # centroids from the decoded order: A gets slot st, B slot 1 - st
        ia = torch.from_numpy(st)
        ar = torch.arange(n)
        za = z[ar, ia] * wt[ar, ia].unsqueeze(-1)
        zb = z[ar, 1 - ia] * wt[ar, 1 - ia].unsqueeze(-1)
        cent = torch.stack([F.normalize(za.sum(0), dim=-1), F.normalize(zb.sum(0), dim=-1)])
    # link confidence: the cheapest flip of any run of windows containing this one (errors come
    # as whole runs flipped where neighbours carry little evidence, not as single windows)
    d = unary[np.arange(n), st] - unary[np.arange(n), 1 - st]  # unary loss of flipping a window
    link = np.where(st[:-1] == st[1:], pair, -pair) * 2  # loss of cutting between t and t + 1
    P = np.concatenate([[0.0], np.cumsum(d)])
    left = np.concatenate([[0.0], link])  # cut before a run starting at a
    right = np.concatenate([link, [0.0]])  # cut after a run ending at b
    M = P[None, 1:] - P[:-1, None] + left[:, None] + right[None, :]  # run [a, b]
    M[np.tril_indices(n, -1)] = np.inf
    A = np.minimum.accumulate(M[:, ::-1], axis=1)[:, ::-1]  # min over b >= t
    conf = np.minimum.accumulate(A, axis=0).diagonal()  # min over a <= t
    for t, w in enumerate(windows):
        w["assign"] = [int(st[t]), 1 - int(st[t])]
        w["margin"] = float(conf[t])
        w["confident"] = w["margin"] >= margin
    return [cent[0], cent[1]]


@torch.no_grad()
def separate(model, x: np.ndarray, mem, device: str, win_s=8.0, hop_s=6.0, batch=8):
    n = len(x)
    win, hop = int(win_s * SR), int(hop_s * SR)
    starts = list(range(0, max(n - win, 0) + 1, hop))
    if starts[-1] + win < n:
        starts.append(max(n - win, 0))
    out = np.zeros((2, n), np.float32)
    wsum = np.zeros(n, np.float32)
    ov = win - hop
    zs = torch.stack(mem).to(device)
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        seg = torch.from_numpy(np.stack([np.pad(x[s:s + win], (0, max(0, win - len(x[s:s + win]))))
                                         for s in ss])).to(device)
        B = len(ss)
        y = model.sep(seg.repeat_interleave(2, 0), zs.repeat(B, 1)).float().view(B, 2, -1).cpu().numpy()
        for s, yy in zip(ss, y, strict=True):
            L = min(win, n - s)
            w = np.ones(L, np.float32)
            if s > 0:
                w[: min(ov, L)] = np.linspace(0, 1, min(ov, L), dtype=np.float32)
            if s + win < n:
                w[-ov:] = np.linspace(1, 0, ov, dtype=np.float32)
            out[:, s:s + L] += yy[:, :L] * w
            wsum[s:s + L] += w
    return out / np.maximum(wsum, 1e-6)


def run(model, x: np.ndarray, device: str):
    windows = identity_pass(model, x, device)
    mem = link_windows(windows)
    if mem[0] is None:
        return np.zeros((2, len(x)), np.float32), {"error": "no confident voice found", "windows": len(windows)}
    y = separate(model, x, mem, device)
    manifest = {
        "sample_rate": SR, "samples": len(x),
        "windows": [{"start_s": w["start"] / SR, "end_s": w["end"] / SR, "assign_slot_of_A_B": w["assign"],
                     "margin": w["margin"], "confident": w["confident"], "solo_s": w["solo_s"]} for w in windows],
        "uncertain_spans_s": [[w["start"] / SR, w["end"] / SR] for w in windows if not w["confident"]],
        "clipping_fraction": float(np.mean(np.abs(y) >= 1.0)),
    }
    manifest["_activity"] = memory_activity(windows, len(x))
    return y, manifest


def memory_activity(windows, n_samples: int, combine: str = "max") -> np.ndarray:
    """Activity of memory slots A and B on the 20 ms grid: each window's slot probabilities
    mapped through its assignment; where windows overlap, their maximum or (``combine="mean"``)
    their average. [2, F20]"""
    nf = n_samples // 320 + 1
    act = np.zeros((2, nf), np.float32)
    cnt = np.zeros(nf, np.float32)
    for w in windows:
        if w["assign"] is None:
            continue
        f0 = w["start"] // 320
        p = w["p"].float().numpy()
        for m, slot in enumerate(w["assign"]):
            seg = p[slot][: nf - f0]
            if combine == "mean":
                act[m, f0:f0 + len(seg)] += seg
            else:
                act[m, f0:f0 + len(seg)] = np.maximum(act[m, f0:f0 + len(seg)], seg)
        cnt[f0:f0 + len(p[0][: nf - f0])] += 1
    return act / np.maximum(cnt, 1) if combine == "mean" else act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly

    x, rate = sf.read(a.input, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if rate != SR:
        g = np.gcd(rate, SR)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)
    model = load(a.model, a.device)
    y, manifest = run(model, x, a.device)
    os.makedirs(a.out_dir, exist_ok=True)
    sf.write(f"{a.out_dir}/speaker_A.wav", y[0], SR)
    sf.write(f"{a.out_dir}/speaker_B.wav", y[1], SR)
    sf.write(f"{a.out_dir}/stereo.wav", y.T, SR)
    manifest.pop("_activity", None)
    json.dump(manifest, open(f"{a.out_dir}/manifest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "windows"}))


if __name__ == "__main__":
    main()
