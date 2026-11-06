"""Where do A/B swaps on whole recordings come from? Reads identity-pass windows cached by
``s1_eval_full.py --cache`` and the Open Yap labels, links the windows and, for every 16 s
window with both voices, compares three orders of its slots: the true one (labels), the linked
one and a split check (first and second half of the window on their own).

    python experiments/K-M1/s1_swap_diag.py --data /data/oy16 --split-file split.json \
        --cache /data/fe/cache_c [--rec] [--link-params '{"emb_w": 4}']

Counts, pooled over conversations:
* link_errors: windows whose own slot order is clear from the labels but linked the other way;
* inner_flips: windows whose two halves prefer different slot orders (the model swaps slots
  inside a window; no linking can fix those);
* conversations with any link error.
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
from m2s_infer import link_windows  # noqa: E402
from m2s_model import frames_20ms  # noqa: E402
from rdata import _Conversation, splits  # noqa: E402


def order_score(p: np.ndarray, act: np.ndarray):
    """(agree keeping slot order, agree crossed) in frames; p, act [2, F] bool."""
    return (float((p[0] & act[0]).sum() + (p[1] & act[1]).sum()),
            float((p[0] & act[1]).sum() + (p[1] & act[0]).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rec", action="store_true", help="windows of rendered recordings")
    ap.add_argument("--conversations", type=int, default=113)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--link-params", default="{}")
    a = ap.parse_args()
    lp = json.loads(a.link_params)
    _, val, _ = splits(a.data, split_file=a.split_file or None)
    tot = {"windows": 0, "clear": 0, "link_errors": 0, "inner_flips": 0, "conv_with_link_error": 0, "conv": 0}
    for pair in val[: a.conversations]:
        f = f"{a.cache}/{pair.conversation_id}{'_rec' if a.rec else ''}.pt"
        if not os.path.exists(f):
            continue
        ws = torch.load(f, weights_only=False)
        link_windows(ws, **lp)
        c = _Conversation(pair)
        nfr = min(c.frames, int(a.max_minutes * 6000))
        act10 = torch.from_numpy(np.stack([c.act[0][:nfr], c.act[1][:nfr]]))
        nf = nfr * 160 // 320 + 1
        act = frames_20ms(act10, nf).bool().numpy()
        # the recording-level order: which memory slot is speaker a
        agree = np.zeros(2)
        rows = []
        for w in ws:
            f0 = w["start"] // 320
            p = (w["p"].float().numpy() > 0.5)
            L = min(p.shape[1], nf - f0)
            p, lab = p[:, :L], act[:, f0:f0 + L]
            if lab[0].sum() < 50 or lab[1].sum() < 50:
                continue
            k, x = order_score(p, lab)
            h = L // 2
            k1, x1 = order_score(p[:, :h], lab[:, :h])
            k2, x2 = order_score(p[:, h:], lab[:, h:])
            rows.append((w["assign"][0], k, x, (k1 - x1), (k2 - x2), lab[:, :h].sum(1).min(), lab[:, h:].sum(1).min()))
            # linked slot of A is assign[0]; true slot of speaker a in this window is 0 if k > x
            agree[0 if (w["assign"][0] == 0) == (k >= x) else 1] += abs(k - x)
        conv_order = 0 if agree[0] >= agree[1] else 1  # 0: memory A is speaker a
        errs = 0
        for slot_a, k, x, d1, d2, m1, m2 in rows:
            tot["windows"] += 1
            if abs(k - x) < 0.3 * (k + x):
                continue
            tot["clear"] += 1
            true_slot_a = 0 if k > x else 1
            linked_slot_a = slot_a if conv_order == 0 else 1 - slot_a
            errs += int(true_slot_a != linked_slot_a)
            if m1 >= 25 and m2 >= 25 and d1 * d2 < 0 and min(abs(d1), abs(d2)) > 25:
                tot["inner_flips"] += 1
        tot["link_errors"] += errs
        tot["conv_with_link_error"] += int(errs > 0)
        tot["conv"] += 1
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
