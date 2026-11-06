"""Where do the swapped 4 s blocks of the full-recording eval come from? Reads identity-pass
windows cached by ``s1_eval_full.py --cache`` and the Open Yap labels, links the windows as the
eval does and, for every 4 s block that ``score_conversation`` counts as swapped, asks each
16 s window that covers it:

* is the window linked the wrong way as a whole (its own slot order, from the labels over the
  whole window, differs from the linked one): a link error;
* is the window linked right but its slots swapped in this block alone: a flip inside the window
  (the activity model changes which slot holds which voice);
* do the covering windows disagree.

A block is "link" when every covering window is linked wrong, "inner" when every covering window
is linked right and swapped in the block, "mixed" otherwise. Also, per block, the share of
overlapped speech and of the minor voice; and, over all windows, how pure the predicted solo of
each slot is (share of its frames where the labels say: its own voice alone, the other voice
alone, both, nobody), for windows linked right and wrong.

    python experiments/K-M1/s1_swap_sources.py --data /data/oy16 --split-file split.json \
        --cache /data/rl/cache [--rec] [--out sources.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
from m2s_infer import link_windows, memory_activity  # noqa: E402
from m2s_model import frames_20ms  # noqa: E402
from rdata import _Conversation, splits  # noqa: E402
import tracker  # noqa: E402

WIN = 200  # 4 s blocks on the 20 ms grid, as in s1_eval_full.score_conversation


def order(p: np.ndarray, lab: np.ndarray) -> tuple[float, float]:
    """(agreement keeping slot order, agreement crossed) in frames; p, lab [2, F] bool."""
    return (float((p[0] & lab[0]).sum() + (p[1] & lab[1]).sum()),
            float((p[0] & lab[1]).sum() + (p[1] & lab[0]).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rec", action="store_true")
    ap.add_argument("--conversations", type=int, default=113)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--link-params", default="{}")
    ap.add_argument("--combine", choices=("max", "mean", "center"), default="max")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lp = json.loads(a.link_params)
    _, val, _ = splits(a.data, split_file=a.split_file or None)
    convs = val[: a.conversations]
    run = tracker.init(f"s1_swap_sources{'_rec' if a.rec else ''}", vars(a), ["kezek-v3", "stage1", "swap-sources"], len(convs))
    kinds, blocks = Counter(), []
    purity = {k: np.zeros(4) for k in ("linked_right", "linked_wrong")}  # own, other, both, none
    n_blocks = 0
    for i, pair in enumerate(convs):
        tracker.log(run, {"progress/swapped_blocks": len(blocks)}, i)
        f = f"{a.cache}/{pair.conversation_id}{'_rec' if a.rec else ''}.pt"
        if not os.path.exists(f):
            continue
        ws = torch.load(f, weights_only=False)
        link_windows(ws, **lp)
        c = _Conversation(pair)
        nfr = min(c.frames, int(a.max_minutes * 6000))
        n_samples = nfr * 160
        p_act = memory_activity(ws, n_samples, a.combine) > 0.5
        nf = p_act.shape[1]
        act = frames_20ms(torch.from_numpy(np.stack([c.act[0][:nfr], c.act[1][:nfr]])), nf).bool().numpy()
        k, x = order(p_act, act)
        mem_of = (0, 1) if k >= x else (1, 0)  # memory slot of speaker a, b
        p = p_act[list(mem_of)]
        # per window: slot activity mapped to speakers a, b through the linked order
        wins = []
        for w in ws:
            f0 = w["start"] // 320
            pw = (w["p"].float().numpy() > 0.5)
            L = min(pw.shape[1], nf - f0)
            if L <= 0:
                continue
            slot_of_mem = w["assign"]  # memory m holds window slot assign[m]
            q = np.stack([pw[slot_of_mem[mem_of[0]], :L], pw[slot_of_mem[mem_of[1]], :L]])
            lab = act[:, f0:f0 + L]
            kk, xx = order(q, lab)
            clear = abs(kk - xx) >= 0.3 * (kk + xx) and lab[0].sum() >= 50 and lab[1].sum() >= 50
            wrong = clear and xx > kk
            wins.append((f0, L, q, wrong, clear))
            if clear:
                solo = np.stack([q[0] & ~q[1], q[1] & ~q[0]])
                key = "linked_wrong" if wrong else "linked_right"
                for s in (0, 1):
                    fr = solo[s]
                    own, oth = lab[s][fr], lab[1 - s][fr]
                    purity[key] += [(own & ~oth).sum(), (~own & oth).sum(), (own & oth).sum(), (~own & ~oth).sum()]
        for b0 in range(0, nf - WIN + 1, WIN):
            lab = act[:, b0:b0 + WIN]
            if lab[0].sum() < 25 or lab[1].sum() < 25:
                continue
            n_blocks += 1
            q = p[:, b0:b0 + WIN]
            keep, swap = (q[0] & lab[0]).sum() + (q[1] & lab[1]).sum(), (q[0] & lab[1]).sum() + (q[1] & lab[0]).sum()
            if not swap > keep + 0.2 * lab.sum():
                continue
            votes = []
            for f0, L, qw, wrong, clear in wins:
                if f0 <= b0 and b0 + WIN <= f0 + L:
                    kb, xb = order(qw[:, b0 - f0:b0 - f0 + WIN], lab)
                    votes.append(("link" if wrong else "inner") if xb > kb else "ok")
            kind = votes[0] if votes and len(set(votes)) == 1 else "mixed"
            kinds[kind] += 1
            both = float((lab[0] & lab[1]).sum() / max((lab[0] | lab[1]).sum(), 1))
            minor = float(lab.sum(1).min() / max(lab.sum(), 1))
            blocks.append({"conv": pair.conversation_id, "t_s": b0 * 0.02, "kind": kind, "votes": votes,
                           "overlap": round(both, 3), "minor": round(minor, 3)})
    out = {"two_speaker_blocks": n_blocks, "swapped_blocks": len(blocks), "kinds": dict(kinds),
           "conversations_with_swaps": len({b["conv"] for b in blocks})}
    for kind in kinds:
        bs = [b for b in blocks if b["kind"] == kind]
        out[f"{kind}_overlap_mean"] = round(float(np.mean([b["overlap"] for b in bs])), 3)
        out[f"{kind}_minor_mean"] = round(float(np.mean([b["minor"] for b in bs])), 3)
    for key, v in purity.items():
        out[f"solo_purity_{key}"] = dict(zip(("own", "other", "both", "none"), np.round(v / max(v.sum(), 1), 3).tolist()))
    print(json.dumps(out))
    tracker.log(run, {f"sources{'_rec' if a.rec else ''}/{k}": v for k, v in {**out, **{f"kind_{k}": v for k, v in kinds.items()}}.items()
                      if isinstance(v, (int, float))}, len(convs))
    tracker.finish(run)
    if a.out:
        json.dump({"summary": out, "blocks": blocks}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
