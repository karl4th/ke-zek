"""Build the real-recording test pack used by kezek.realtest.

Input: a JSON list of recordings, each {"id", "wav" (16 kHz mono), "source", "title", "start_s",
"duration_s", optional "enroll_s": [start, end] (a hand-checked stretch of one speaker, used as
that speaker's voice sample instead of the automatic one)}. Output folder: manifest.json,
audio/<id>.flac, labels/<id>.json.

Labels: ECAPA embeddings of 1.5 s windows every 0.5 s (quiet windows skipped), cosine k-means
into two speakers; a 0.5 s chunk is labeled only when its window is closer to one centroid by a
margin and the two windows on each side agree, so pauses, turn edges and overlaps stay -1.
Voice sample per speaker: its longest labeled stretches, 30 s.

    python experiments/realtest/build_pack.py recordings.json out_dir
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from kezek.model import SpeakerEncoder

SR, WIN, HOP = 16000, int(1.5 * 16000), 8000
MARGIN, ENROLL_S = 0.10, 30.0


def kmeans(x: np.ndarray, k: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    c = x[[0]]
    for _ in range(1, k):  # farthest-point init
        c = np.vstack([c, x[int(np.argmax(1 - (x @ c.T).max(1)))]])
    for _ in range(iters):
        lab = np.argmax(x @ c.T, 1)
        c = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        c /= np.linalg.norm(c, axis=1, keepdims=True)
    return lab, c


@torch.no_grad()
def split(x: np.ndarray, se) -> dict:
    starts = np.arange(0, len(x) - WIN, HOP)
    rms = np.array([10 * np.log10(np.mean(x[s : s + WIN] ** 2) + 1e-10) for s in starts])
    keep = rms > np.percentile(rms, 95) - 30
    emb = np.zeros((len(starts), 192), np.float32)
    idx = np.flatnonzero(keep)
    dev = next(se.parameters()).device
    for i in range(0, len(idx), 64):
        b = idx[i : i + 64]
        w = torch.from_numpy(np.stack([x[s : s + WIN] for s in starts[b]])).to(dev)
        emb[b] = se(w).cpu().numpy()
    e = emb[keep] / np.linalg.norm(emb[keep], axis=1, keepdims=True)
    lab, c = kmeans(e, 2)
    sims = e @ c.T
    sure = np.abs(sims[:, 0] - sims[:, 1]) > MARGIN
    w_lab = np.full(len(starts), -1)
    w_lab[idx[sure]] = lab[sure]
    chunk = np.full(len(x) // HOP, -1)
    for i in range(2, len(starts) - 2):
        if w_lab[i] >= 0 and (w_lab[i - 2 : i + 3] == w_lab[i]).all():
            chunk[(starts[i] + WIN // 2) // HOP] = w_lab[i]
    return {"chunk_s": HOP / SR, "chunks": chunk.tolist(), "centroid_cos": float(c[0] @ c[1])}


def enrollment(chunk: np.ndarray, spk: int, need: int) -> list[list[int]]:
    m = np.concatenate([[False], chunk == spk, [False]])
    d = np.flatnonzero(np.diff(m.astype(int)))
    runs = sorted(zip(d[::2], d[1::2], strict=True), key=lambda ab: ab[0] - ab[1])
    used, total = [], 0
    for a, b in runs:
        take = min(b - a, need - total)
        used.append([int(a), int(a + take)])
        total += take
        if total >= need:
            break
    return used


def main():
    recs, out = json.load(open(sys.argv[1])), sys.argv[2]
    os.makedirs(f"{out}/audio", exist_ok=True)
    os.makedirs(f"{out}/labels", exist_ok=True)
    se = SpeakerEncoder(device="cuda")
    need = int(ENROLL_S * SR / HOP)
    items = []
    for r in recs:
        x, sr = sf.read(r["wav"], dtype="float32")
        assert sr == SR and x.ndim == 1, r["wav"]
        lab = split(x, se)
        chunk = np.asarray(lab["chunks"])
        lab["speakers"] = {}
        for spk in (0, 1):
            enr = enrollment(chunk, spk, need)
            lab["speakers"][str(spk)] = {
                "labeled_s": float((chunk == spk).sum() * HOP / SR),
                "enrollment_chunks": enr,
                "enrollment_s": sum(b - a for a, b in enr) * HOP / SR,
            }
        if r.get("enroll_s"):  # a hand-checked stretch replaces that speaker's automatic sample
            a, b = int(r["enroll_s"][0] * SR / HOP), int(r["enroll_s"][1] * SR / HOP)
            spk = int(np.bincount(chunk[a:b][chunk[a:b] >= 0], minlength=2).argmax())
            lab["speakers"][str(spk)]["enrollment_chunks"] = [[a, b]]
            lab["speakers"][str(spk)]["enrollment_s"] = (b - a) * HOP / SR
            lab["speakers"][str(spk)]["enrollment_note"] = "hand-checked stretch"
        sf.write(f"{out}/audio/{r['id']}.flac", x, SR, subtype="PCM_16")
        json.dump(lab, open(f"{out}/labels/{r['id']}.json", "w"))
        sha = hashlib.sha256(open(f"{out}/audio/{r['id']}.flac", "rb").read()).hexdigest()
        items.append(
            {
                "id": r["id"],
                "source": r["source"],
                "title": r["title"],
                "start_s": r["start_s"],
                "duration_s": len(x) / SR,
                "audio": f"audio/{r['id']}.flac",
                "labels": f"labels/{r['id']}.json",
                "audio_sha256": sha,
            }
        )
        s0, s1 = lab["speakers"]["0"], lab["speakers"]["1"]
        print(
            f"{r['id']}: labeled {(chunk >= 0).mean():.0%} | s0 {s0['labeled_s'] / 60:.1f} min "
            f"s1 {s1['labeled_s'] / 60:.1f} min | centroid cos {lab['centroid_cos']:.2f}",
            flush=True,
        )
    json.dump(
        {
            "version": 1,
            "sample_rate": SR,
            "items": items,
            "note": "YouTube podcasts, internal testing only; automatic labels (ECAPA)",
        },
        open(f"{out}/manifest.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
