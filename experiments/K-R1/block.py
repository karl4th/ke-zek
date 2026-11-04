"""K-R1 block, stage 2: pick the target's speech out of two separated streams of a conversation.

The separator gives two streams but does not know who is who, and its order may change from
window to window. The streams are cut into speech runs, and every run is given to the target or
to the other person under two kinds of evidence:

* voice: the cosine between the run's ECAPA embedding and the enrollment, against a threshold
  set per recording between the two clusters of run scores (two people, two clusters);
* conversation: two runs that sound at the same time in different streams belong to different
  people. A short "mhm" under the other person's long turn has too little voice for ECAPA, but
  the long turn it overlaps is easy to place, and the reply takes the other label.

Runs linked by overlaps form components; each component has two consistent labelings, and the
one that agrees best with the voice evidence (weighted by run duration) wins.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 16_000
FRAME = 160


def separate_long(sep, mix: np.ndarray, device: str, win_s: float = 8.0, hop_s: float = 6.0):
    """Two streams over a whole recording: windows with a fixed hop, the order of each window
    matched to the previous one on their overlap (correlation), linear crossfade."""
    win, hop = int(win_s * SR), int(hop_s * SR)
    x = np.asarray(mix, np.float32)
    n = len(x)
    if n <= win:
        with torch.no_grad():
            return sep(torch.from_numpy(x)[None].to(device))[0].float().cpu().numpy()
    starts = list(range(0, n - win, hop)) + [n - win]
    out = np.zeros((2, n), np.float32)
    wsum = np.zeros(n, np.float32)
    prev = None
    for s in starts:
        with torch.no_grad():
            y = sep(torch.from_numpy(x[s : s + win])[None].to(device))[0].float().cpu().numpy()
        if prev is not None:
            ps, py = prev
            a, b = s, min(ps + win, s + win)
            if b > a:
                p = py[:, a - ps : b - ps]
                q = y[:, : b - a]
                keep = (p[0] @ q[0] + p[1] @ q[1]) >= (p[0] @ q[1] + p[1] @ q[0])
                if not keep:
                    y = y[::-1].copy()
        w = np.ones(win, np.float32)
        ov = win - hop
        if s > 0:
            w[:ov] = np.linspace(0, 1, ov, dtype=np.float32)
        if s + win < n:
            w[-ov:] = np.linspace(1, 0, ov, dtype=np.float32)
        out[:, s : s + win] += y * w
        wsum[s : s + win] += w
        prev = (s, y)
    return out / np.maximum(wsum, 1e-6)


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    return 10 * np.log10(np.mean(np.float64(x[: n * FRAME]).reshape(n, FRAME) ** 2, axis=1) + 1e-12)


def stream_runs(y: np.ndarray, ref_db: float, rel_db: float = 35.0, merge: int = 30, min_len: int = 10):
    """Speech runs of one stream (frame index pairs): frames within ``rel_db`` of the mixture's
    loud level, 100 ms hangover, gaps under ``merge`` frames closed."""
    act = frame_db(y) > ref_db - rel_db
    k = np.ones(21)
    act = np.convolve(act.astype(float), k, mode="same") > 0
    idx = np.flatnonzero(act)
    for i, j in zip(idx[:-1], idx[1:], strict=False):
        if 1 < j - i <= merge:
            act[i:j] = True
    p = np.concatenate([[False], act, [False]])
    e = np.flatnonzero(p[1:] != p[:-1]).reshape(-1, 2)
    return [(int(a), int(b)) for a, b in e if b - a >= min_len]


def two_means_threshold(scores: np.ndarray, weights: np.ndarray) -> float:
    """Midpoint between the two weighted 1-D clusters of run scores."""
    if len(scores) < 2:
        return float(scores.mean()) if len(scores) else 0.0
    lo, hi = float(scores.min()), float(scores.max())
    for _ in range(50):
        t = 0.5 * (lo + hi)
        a, b = scores <= t, scores > t
        if not a.any() or not b.any():
            break
        nlo = float(np.average(scores[a], weights=weights[a]))
        nhi = float(np.average(scores[b], weights=weights[b]))
        if abs(nlo - lo) < 1e-6 and abs(nhi - hi) < 1e-6:
            break
        lo, hi = nlo, nhi
    return 0.5 * (lo + hi)


@torch.no_grad()
def assign(
    streams: np.ndarray,
    mix: np.ndarray,
    enr_embedding: torch.Tensor,
    se,
    device: str,
    use_conversation: bool = True,
    min_voice_s: float = 0.5,
    min_overlap: int = 10,
    rel_db: float = 35.0,
    refine: int = 5,
    min_target_cos: float = 0.35,
    gate_frames: int = 200,
    same_cos: float = 0.5,
    fade: int = 160,
    oracle: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """Target waveform from two separated streams; ``info`` describes the decision.

    With ``refine`` > 0 the voice evidence of a run is the difference of its cosines to the two
    people's centroids (each the duration-weighted mean of the runs currently given to them, the
    target's anchored by the enrollment), and labels and centroids are re-estimated that many
    times. With 0, the cosine to the enrollment against a two-cluster threshold."""
    ref_db = float(np.percentile(frame_db(mix), 95))
    runs = [(s, a, b) for s in (0, 1) for a, b in stream_runs(streams[s], ref_db, rel_db)]
    n = len(runs)
    if n == 0:
        return np.zeros_like(mix), {"runs": 0}
    score = np.zeros(n)
    weight = np.zeros(n)
    emb = np.zeros((n, enr_embedding.shape[-1]), np.float32)
    enr = torch.nn.functional.normalize(enr_embedding, dim=-1)[0].cpu().numpy()
    for i, (s, a, b) in enumerate(runs):
        if (b - a) * 0.01 < min_voice_s:
            continue
        seg = streams[s][a * FRAME : min(b, a + 1000) * FRAME]
        e = se(torch.from_numpy(np.ascontiguousarray(seg, np.float32))[None].to(device))
        emb[i] = torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()
        score[i] = float(emb[i] @ enr)
        weight[i] = min((b - a) * 0.01, 5.0)
    voiced = weight > 0
    thr = two_means_threshold(score[voiced], weight[voiced]) if voiced.any() else 0.0
    evidence = np.where(voiced, weight * (score - thr), 0.0)

    # components of runs linked by "different person" edges, with parity
    parent, parity = list(range(n)), [0] * n

    def find(i):
        if parent[i] == i:
            return i, 0
        r, p = find(parent[i])
        parent[i], parity[i] = r, parity[i] ^ p
        return r, parity[i]

    edges = []
    if use_conversation:
        order = sorted(range(n), key=lambda i: runs[i][1])
        for k, i in enumerate(order):
            si, ai, bi = runs[i]
            for j in order[k + 1 :]:
                sj, aj, bj = runs[j]
                if aj >= bi:
                    break
                ov = min(bi, bj) - aj
                if si != sj and ov >= min_overlap:
                    edges.append((ov, i, j))
    edges.sort(reverse=True)
    conflicts = 0
    for _, i, j in edges:
        ri, pi = find(i)
        rj, pj = find(j)
        if ri == rj:
            conflicts += int(pi == pj)
            continue
        parent[rj], parity[rj] = ri, pi ^ pj ^ 1
    roots = [find(i) for i in range(n)]

    def labels_from(ev):
        comp: dict[int, float] = {}
        for i, (r, p) in enumerate(roots):
            comp[r] = comp.get(r, 0.0) + (ev[i] if p == 0 else -ev[i])
        lab = np.zeros(n, bool)
        for i, (r, p) in enumerate(roots):
            t = comp[r] > 0
            lab[i] = t if p == 0 else not t
        return lab

    label = labels_from(evidence)
    for _ in range(refine if voiced.sum() >= 2 else 0):
        wt = weight * label
        wo = weight * ~label
        mu_t = enr * 2.0 + (wt[:, None] * emb).sum(0)
        mu_o = (wo[:, None] * emb).sum(0)
        if not wo.any():
            break
        mu_t /= np.linalg.norm(mu_t) + 1e-9
        mu_o /= np.linalg.norm(mu_o) + 1e-9
        ev = np.where(voiced, weight * (emb @ mu_t - emb @ mu_o), 0.0)
        new = labels_from(ev)
        if (new == label).all():
            break
        label = new
    # Two clusters always exist, even when only one person talks in this stretch. If the long
    # runs on the two sides sound like the same person (centroid cosine >= ``same_cos``), there
    # is one voice here, and it is the target only if it matches the enrollment in absolute
    # terms (ECAPA cosine of runs >= 2 s: same person median 0.62, other person p95 0.26).
    # ``gated``: 0 two voices, 1 one voice = target, -1 one voice = the other person.
    gated = 0
    long_runs = np.array([b - a >= gate_frames for _, a, b in runs]) & voiced
    if min_target_cos > -1 and long_runs.any():
        lt, lo = long_runs & label, long_runs & ~label

        def centroid(m):
            v = (weight[m][:, None] * emb[m]).sum(0)
            return v / (np.linalg.norm(v) + 1e-9)

        one_voice = not lt.any() or not lo.any() or float(centroid(lt) @ centroid(lo)) >= same_cos
        if one_voice:
            # the long runs are all this one person; short replies overlapping them take the other
            # label through the conversation edges
            sign = 1.0 if float(centroid(long_runs) @ enr) >= min_target_cos else -1.0
            label = labels_from(np.where(long_runs, sign * 1e3 * weight, evidence))
            gated = 1 if sign > 0 else -1
    if oracle is not None:  # diagnostics: each run goes to whichever clean track it is closer to
        t_ref, o_ref = oracle
        for i, (s_, a_, b_) in enumerate(runs):
            seg = streams[s_][a_ * FRAME : b_ * FRAME]
            label[i] = float(np.sum((seg - t_ref[a_ * FRAME : b_ * FRAME]) ** 2)) < float(
                np.sum((seg - o_ref[a_ * FRAME : b_ * FRAME]) ** 2)
            )
    out = np.zeros(len(mix), np.float32)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    for (s, a, b), t in zip(runs, label, strict=True):
        if not t:
            continue
        x0, x1 = a * FRAME, min(b * FRAME, len(mix))
        w = np.ones(x1 - x0, np.float32)
        k = min(fade, len(w) // 2)
        if k:
            w[:k] *= ramp[:k]
            w[-k:] *= ramp[:k][::-1]
        out[x0:x1] += streams[s][x0:x1] * w
    info = {
        "runs": n,
        "voiced_runs": int(voiced.sum()),
        "edges": len(edges),
        "conflicts": conflicts,
        "threshold": thr,
        "target_runs": int(label.sum()),
        "gated_components": gated,
    }
    return out, info
