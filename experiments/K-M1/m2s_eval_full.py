"""Full-conversation evaluation of the two-block Kezek (M2S v0.1 sections 3 and 6).

    python experiments/K-M1/m2s_eval_full.py --data /data/oy16 --split-file split.json \
        --model run/model.pt --split val --conversations 12 --out val.json
    ... --split test --final

Whole held-out Open Yap conversations (up to ``--max-minutes`` from the start), mono-only
inference with slot memory (m2s_infer.run). The A/B mapping for scoring is chosen ONCE per
conversation (the order with the larger summed SI-SDR over the whole recording) and never per
window. Reported:
* segment metrics per speaker (K-T1 definitions) pooled over both outputs and all conversations;
* SI-SDRi over 4 s windows where the speaker is active: overall, overlap-only, single-speaker;
* swaps: 4 s windows where both speak >= 0.5 s and the other mapping is better by > 3 dB
  summed SI-SDR; per conversation and per hour; windows flagged uncertain by the memory are
  counted separately;
* activity precision / recall / F1 from the identity pass (mapped to the canonical order),
  sample count check and clipping.
The test split is read only with --final.
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
import tracker  # noqa: E402
from m2s_infer import load, run  # noqa: E402
from rdata import _Conversation, disputed_frames, splits  # noqa: E402

from kezek import metrics  # noqa: E402

SR, FRAME = 16_000, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--conversations", type=int, default=12)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        raise SystemExit("the test split is read only with --final")
    _, val, test = splits(a.data, split_file=a.split_file or None)
    pairs = (val if a.split == "val" else test)[: a.conversations]
    model = load(a.model, a.device)
    scores, gains, gains_ov, gains_solo, per_conv = [], [], [], [], []
    tp = fp = fn = 0.0
    for pair in pairs:
        c = _Conversation(pair)
        nfr = min(c.frames, int(a.max_minutes * 6000))
        sa, sb = c.audio(0, 0, nfr), c.audio(1, 0, nfr)
        x = sa + sb
        g = 0.9 / max(float(np.abs(x).max()), 1e-4)
        sa, sb, x = sa * g, sb * g, x * g
        y, man = run(model, x, a.device)
        assert y.shape[-1] == len(x), "output length differs from input"
        keep = metrics.si_sdr(y[0], sa) + metrics.si_sdr(y[1], sb)
        swap = metrics.si_sdr(y[0], sb) + metrics.si_sdr(y[1], sa)
        order = (0, 1) if keep >= swap else (1, 0)  # output index for source a, b
        src = (sa, sb)
        act = (c.act[0][:nfr], c.act[1][:nfr])
        disp = disputed_frames(act[0], act[1])
        for s in (0, 1):
            m = np.repeat(act[s], FRAME)
            l_ref = float(np.mean(np.float64(src[s][m]) ** 2)) if m.any() else 1e-8
            scores.append(metrics.segment_scores(y[order[s]], src[s], src[1 - s], x, act[s], act[1 - s], disp, l_ref))
        n_sw = n_win = n_sw_flag = 0
        win = 400
        uncertain = np.zeros(nfr, bool)
        for lo, hi in man.get("uncertain_spans_s", []):
            uncertain[int(lo * 100):int(hi * 100)] = True
        for f0 in range(0, nfr - win + 1, win):
            sl = slice(f0 * FRAME, (f0 + win) * FRAME)
            a0, a1 = act[0][f0:f0 + win].sum(), act[1][f0:f0 + win].sum()
            for s, ac, oc in ((0, a0, a1), (1, a1, a0)):
                if ac >= 50:
                    gain = metrics.si_sdr(y[order[s]][sl], src[s][sl]) - metrics.si_sdr(x[sl], src[s][sl])
                    gains.append(gain)
                    (gains_ov if oc >= 50 else gains_solo).append(gain)
            if a0 >= 50 and a1 >= 50:
                k_ = metrics.si_sdr(y[order[0]][sl], sa[sl]) + metrics.si_sdr(y[order[1]][sl], sb[sl])
                s_ = metrics.si_sdr(y[order[1]][sl], sa[sl]) + metrics.si_sdr(y[order[0]][sl], sb[sl])
                n_win += 1
                if s_ > k_ + 3.0:
                    n_sw += 1
                    n_sw_flag += int(uncertain[f0:f0 + win].any())
        per_conv.append({"conv": pair.conversation_id, "minutes": nfr / 6000, "swap_windows": n_sw,
                         "swap_windows_flagged": n_sw_flag, "two_speaker_windows": n_win,
                         "uncertain_windows": len(man.get("uncertain_spans_s", [])),
                         "clipping": man.get("clipping_fraction", 0.0)})
        # activity of memory slots A/B (20 ms) against the sources, in the canonical order
        pred = man["_activity"] > 0.5
        for s in (0, 1):
            lab = act[s][: (len(act[s]) // 2) * 2].reshape(-1, 2).max(1)
            pr = pred[order[s]][: len(lab)]
            lab = lab[: len(pr)]
            tp += float((pr & lab).sum())
            fp += float((pr & ~lab).sum())
            fn += float((~pr & lab).sum())
        print(json.dumps(per_conv[-1]), flush=True)
    out = {"split": a.split, "model": a.model, "conversations": len(per_conv), **metrics.aggregate(scores)}
    prec, rec = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    out.update({"act_precision": prec, "act_recall": rec, "act_f1": 2 * prec * rec / max(prec + rec, 1e-9)})
    hours = sum(p["minutes"] for p in per_conv) / 60
    sw = sum(p["swap_windows"] for p in per_conv)
    out.update({
        "sisdri_overall": float(np.mean(gains)) if gains else float("nan"),
        "sisdri_overlap_windows": float(np.mean(gains_ov)) if gains_ov else float("nan"),
        "sisdri_single_windows": float(np.mean(gains_solo)) if gains_solo else float("nan"),
        "swap_windows": sw, "swap_windows_per_hour": sw / max(hours, 1e-9),
        "swap_windows_unflagged": sw - sum(p["swap_windows_flagged"] for p in per_conv),
        "conversations_with_swaps": sum(p["swap_windows"] > 0 for p in per_conv), "hours": hours,
        "per_conversation": per_conv,
    })
    json.dump(out, open(a.out, "w"), indent=1)
    name = f"k-m1-full-{a.split}-{os.path.splitext(os.path.basename(a.out))[0]}"
    os.environ.setdefault("AIRON_EXPERIMENT", "K-M1 two-block Kezek on Open Yap")
    r = tracker.init(name, {"model": a.model, "split": a.split, "conversations": len(per_conv),
                            "max_minutes": a.max_minutes}, ["full-conversation", a.split], 1)
    tracker.log(r, {k: v for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}, 0)
    tracker.finish(r)
    show = ("overlap_sisdri", "sisdri_overall", "sisdri_overlap_windows", "target_solo_sisdr",
            "other_solo_leak_db", "backchannel_recall", "swap_windows", "swap_windows_per_hour",
            "conversations_with_swaps", "act_f1")
    print(json.dumps({k: round(out[k], 3) if isinstance(out[k], float) else out[k] for k in show}))


if __name__ == "__main__":
    main()
