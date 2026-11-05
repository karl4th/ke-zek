"""Stage 1 activity on held-out spans at several decision thresholds: precision, recall, F1,
short-reply recall and false overlap (frames with exactly one person talking where both slots
fire), with and without frames near activity edges (pyannote-free Open Yap labels).

    python experiments/K-M1/s1_thresholds.py --data /data/oy16 --split-file split.json \
        --model run/model.pt [--aug-root /data/aug] [--n 256]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
from m2s_data import SpanSequences, collate  # noqa: E402
from m2s_infer import load  # noqa: E402
from m2s_model import frames_20ms  # noqa: E402
from rdata import _runs, splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--aug-root", default="")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--single-p", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    _, val, _ = splits(a.data, split_file=a.split_file)
    render = None
    if a.aug_root:
        from render import Render

        from kezek.augment import DomainBank

        r = a.aug_root
        render = Render(DomainBank.from_dirs(
            [f"{r}/RIRS_NOISES/simulated_rirs", f"{r}/RIRS_NOISES/real_rirs_isotropic_noises"],
            [f"{r}/musan/noise"], [f"{r}/musan/music"]))
    ds = SpanSequences(val, a.n, seed=21_000, span_s=16, gain_db=0.0, peak=(0.9, 0.9), render=render,
                       single_p=a.single_p)
    m = load(a.model, a.device)
    probs, labs, bcs, disp = [], [], [], []
    with torch.no_grad():
        for i0 in range(0, a.n, 8):
            b = collate([ds[i] for i in range(i0, min(i0 + 8, a.n))])
            logits, _ = m.enc(b["mix"].to(a.device))
            p = torch.sigmoid(logits).cpu()
            nf = p.shape[-1]
            lab = frames_20ms(b["act"], nf).bool()
            bc = frames_20ms(b["bc"], nf).bool()
            dd = frames_20ms(b["disputed"][:, None], nf)[:, 0].bool()
            for j in range(p.shape[0]):  # slot order chosen by agreement at threshold 0.5
                q = p[j] > 0.5
                keep = (q & lab[j]).sum() >= (q & lab[j].flip(0)).sum()
                probs.append(p[j] if keep else p[j].flip(0))
                labs.append(lab[j])
                bcs.append(bc[j])
                disp.append(dd[j])
    P, L, BC, D = (torch.stack(x).numpy() for x in (probs, labs, bcs, disp))
    one = L.sum(1) == 1
    res = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        q = P > t
        tp, fp, fn = (q & L).sum(), (q & ~L).sum(), (~q & L).sum()
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        both = q.sum(1) == 2
        hit = n = 0
        for j in range(len(P)):
            for s in (0, 1):
                for r0, r1 in _runs(BC[j, s]):
                    n += 1
                    hit += int(q[j, s, r0:r1].mean() >= 0.5)
        row = {"threshold": t, "precision": float(pr), "recall": float(rc), "f1": float(2 * pr * rc / max(pr + rc, 1e-9)),
               "bc_recall": hit / max(n, 1), "false_overlap": float(both[one].mean()),
               "false_overlap_no_edges": float(both[one & ~D].mean())}
        res.append(row)
        print(json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
    print("SHORT REPLY RUNS", n)


if __name__ == "__main__":
    main()
