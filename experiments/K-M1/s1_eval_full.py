"""Kezek v3 stage 1 on whole recordings: identity pass in windows, slot memory, activity of
memory slots A and B (m2s_infer), scored against the Open Yap labels.

    python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file split.json \
        --model run/model.pt --conversations 40 --out s1_val.json [--real-clip clip.wav]

The A/B mapping is chosen ONCE per conversation (the order with more agreeing active frames).
Reported, pooled over conversations:
* activity precision / recall / F1 of the memory slots (20 ms frames, threshold 0.5);
* short-reply recall: runs < 1 s next to the other person's speech, found when >= 50% of their
  frames are active in the right slot;
* swaps: 4 s windows where both speak >= 0.5 s and the swapped mapping agrees on more active
  frames (by a margin of 20% of the window's speech); per conversation, per hour and the share of
  conversations without any; windows the memory flags uncertain are counted separately;
* with ``--real-clip`` (one person talking, no labels): share of speech frames in the dominant
  slot and in the other one.
The test split is read only with --final.
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
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
import tracker  # noqa: E402
from m2s_data import backchannel_frames  # noqa: E402
from m2s_infer import build_memory, identity_pass, link_windows, load, memory_activity, relink_segments  # noqa: E402
from m2s_model import frames_20ms  # noqa: E402
from rdata import _Conversation, _runs, splits  # noqa: E402

SR, FRAME = 16_000, 160


def score_conversation(p_act: np.ndarray, act: np.ndarray, bc: np.ndarray, uncertain: np.ndarray):
    """p_act [2, F] memory activity (bool), act/bc [2, F] labels on the same 20 ms grid."""
    agree = [(p_act[0] & act[0]).sum() + (p_act[1] & act[1]).sum(),
             (p_act[0] & act[1]).sum() + (p_act[1] & act[0]).sum()]
    order = (0, 1) if agree[0] >= agree[1] else (1, 0)  # memory slot of speaker a, b
    p = p_act[list(order)]
    r = {"tp": float((p & act).sum()), "fp": float((p & ~act).sum()), "fn": float((~p & act).sum())}
    hit = hit_any = n = 0
    for s in (0, 1):
        for r0, r1 in _runs(bc[s]):
            n += 1
            hit += int(p[s, r0:r1].mean() >= 0.5)
            hit_any += int((p[0, r0:r1] | p[1, r0:r1]).mean() >= 0.5)
    r.update({"bc_hit": hit, "bc_n": n, "bc_hit_any": hit_any})
    win = 200  # 4 s on the 20 ms grid
    sw = sw_flag = n_win = 0
    for f0 in range(0, act.shape[1] - win + 1, win):
        a = act[:, f0:f0 + win]
        if a[0].sum() < 25 or a[1].sum() < 25:
            continue
        q = p[:, f0:f0 + win]
        keep = (q[0] & a[0]).sum() + (q[1] & a[1]).sum()
        swap = (q[0] & a[1]).sum() + (q[1] & a[0]).sum()
        n_win += 1
        if swap > keep + 0.2 * a.sum():
            sw += 1
            sw_flag += int(uncertain[f0:f0 + win].any())
    r.update({"swap_windows": sw, "swap_windows_flagged": sw_flag, "two_speaker_windows": n_win})
    return r


@torch.no_grad()
def reembed(windows, x, spk, device, lab, mode, thr):
    """Slot embeddings from packed solo audio of each window, by the predicted solo (p_a (1 - p_b)
    above ``thr``) or by the labels (the speaker a slot holds: the one its activity agrees with in
    that window). ``label_full`` also sets the slot activity to the labels."""
    nf = lab.shape[1]
    for w in windows:
        f0 = w["start"] // 320
        p = w["p"].float().numpy()
        L = min(p.shape[1], nf - f0)
        if mode == "pred":
            solo = np.stack([p[0] * (1 - p[1]) > thr, p[1] * (1 - p[0]) > thr])[:, :L]
        else:
            l = lab[:, f0:f0 + L]
            q = p[:, :L] > 0.5
            keep = (q[0] & l[0]).sum() + (q[1] & l[1]).sum()
            cross = (q[0] & l[1]).sum() + (q[1] & l[0]).sum()
            sp = (0, 1) if keep >= cross else (1, 0)  # speaker held by slot 0, 1
            solo = np.stack([l[sp[0]] & ~l[sp[1]], l[sp[1]] & ~l[sp[0]]])
            if mode == "label_full":
                full = np.zeros_like(p)
                full[:, :L] = np.stack([l[sp[0]], l[sp[1]]])
                w["p"] = torch.from_numpy(full)
        rows, secs = [], []
        for s in (0, 1):
            fr = np.flatnonzero(solo[s]) + f0
            idx = (fr[:, None] * 320 + np.arange(320)[None]).ravel()
            idx = idx[idx < len(x)]
            secs.append(len(fr) * 0.02)
            if len(idx) < 1600:  # < 0.1 s: the masked window as in m2s_model._packed
                seg = np.zeros(int(w["end"] - w["start"]), np.float32)
                m = np.repeat(solo[s], 320)[: len(seg)]
                seg[: len(m)] = x[w["start"]:w["start"] + len(m)] * m
                rows.append(seg)
            else:
                rows.append(x[idx])
        Lm = max(len(r) for r in rows)
        batch = torch.stack([F.pad(torch.from_numpy(np.ascontiguousarray(r)), (0, Lm - len(r))) for r in rows]).to(device)
        rel = torch.tensor([len(r) / Lm for r in rows], device=device)
        w["z"] = F.normalize(spk(batch.float(), rel).float().cpu(), dim=-1)
        w["solo_s"] = secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--conversations", type=int, default=40)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--win-s", type=float, default=16.0)
    ap.add_argument("--hop-s", type=float, default=8.0)
    ap.add_argument("--aug-root", default="", help="render each conversation as one mono recording")
    ap.add_argument("--real-clip", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--linker", choices=("memory", "global"), default="global",
                    help="greedy slot memory or global Viterbi linking (m2s_infer.link_windows)")
    ap.add_argument("--cache", default="", help="directory of identity-pass windows per conversation "
                    "(written on the first run, reused after: relinking needs no model)")
    ap.add_argument("--combine", choices=("max", "mean", "center"), default="max",
                    help="overlapping windows: maximum or average of slot activity")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--link-params", default="{}", help="JSON keyword arguments of link_windows")
    ap.add_argument("--relink", default="", help="JSON keyword arguments of relink_segments (second "
                    "linking pass on long solo audio per segment); empty: off")
    ap.add_argument("--tag", default="", help="suffix of the tracker run name")
    ap.add_argument("--reembed", choices=("", "pred", "label", "label_full"), default="",
                    help="slot embeddings again from packed solo audio before linking: 'pred' from the "
                    "predicted solo above --solo-thr, 'label' from the labelled solo of the speaker each "
                    "slot holds (oracle embeddings), 'label_full' also replaces the slot activity by the labels")
    ap.add_argument("--solo-thr", type=float, default=0.5)
    a = ap.parse_args()
    lp = json.loads(a.link_params)
    link = build_memory if a.linker == "memory" else (lambda ws: link_windows(ws, **lp))
    if a.split == "test" and not a.final:
        raise SystemExit("the test split is read only with --final")
    _, val, test = splits(a.data, split_file=a.split_file or None)
    pairs = (val if a.split == "val" else test)[: a.conversations]
    model = load(a.model, a.device)
    render = None
    if a.aug_root:
        sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
        from render import Render

        from kezek.augment import DomainBank

        r = a.aug_root
        render = Render(DomainBank.from_dirs(
            [f"{r}/RIRS_NOISES/simulated_rirs", f"{r}/RIRS_NOISES/real_rirs_isotropic_noises"],
            [f"{r}/musan/noise"], [f"{r}/musan/music"]))
    sfx = "_rec" if render is not None else ""
    variant = ("_pack" if os.environ.get("KEZEK_PACK_SOLO") == "1" else "") + ("_ecapaft" if os.environ.get("KEZEK_ECAPA") else "")
    run = tracker.init(f"s1_full_{a.split}{sfx}_{os.path.basename(os.path.dirname(a.model))}{variant}{a.tag}",
                       {**vars(a), "pack_solo": os.environ.get("KEZEK_PACK_SOLO", "0"), "ecapa": os.environ.get("KEZEK_ECAPA", "")},
                       ["kezek-v3", "stage1", "full-recording", a.split], len(pairs))
    per_conv, hours = [], 0.0
    for k, pair in enumerate(pairs):
        c = _Conversation(pair)
        nfr = min(c.frames, int(a.max_minutes * 6000))
        sa, sb = c.audio(0, 0, nfr), c.audio(1, 0, nfr)
        act10 = np.stack([c.act[0][:nfr], c.act[1][:nfr]])
        if render is not None:
            rng = np.random.default_rng([30_000, k])
            rec = render.sample(rng)
            _, _, x = render.apply(rec, sa, sb, np.repeat(act10[0] | act10[1], FRAME)[: len(sa)], rng)
        else:
            x = sa + sb
        x = (x * np.float32(0.9 / max(float(np.abs(x).max()), 1e-4))).astype(np.float32)
        cf = f"{a.cache}/{pair.conversation_id}{'_rec' if render is not None else ''}.pt" if a.cache else ""
        if cf and os.path.exists(cf):
            windows = torch.load(cf)
        else:
            windows = identity_pass(model, x, a.device, win_s=a.win_s, hop_s=a.hop_s)
            if cf:
                os.makedirs(a.cache, exist_ok=True)
                torch.save(windows, cf)
        if a.reembed:
            nf0 = len(x) // 320 + 1
            reembed(windows, x, model.enc._speaker[0], a.device, frames_20ms(torch.from_numpy(act10), nf0).bool().numpy(),
                    a.reembed, a.solo_thr)
        link(windows)
        if a.relink:
            relink_segments(windows, x, model.enc._speaker[0], a.device, **json.loads(a.relink))
        p_act = memory_activity(windows, len(x), a.combine) > a.threshold
        nf = p_act.shape[1]
        bc10 = np.stack([backchannel_frames(act10[0], act10[1]), backchannel_frames(act10[1], act10[0])])
        act = frames_20ms(torch.from_numpy(act10), nf).bool().numpy()
        bc = frames_20ms(torch.from_numpy(bc10), nf).bool().numpy()
        unc = np.zeros(nf, bool)
        for w in windows:
            if not w["confident"]:
                unc[w["start"] // 320:w["end"] // 320] = True
        r = score_conversation(p_act, act, bc, unc)
        r.update({"conv": pair.conversation_id, "minutes": nfr / 6000,
                  "uncertain_windows": int(sum(not w["confident"] for w in windows)), "windows": len(windows)})
        per_conv.append(r)
        hours += nfr / 360_000
        print(json.dumps(r), flush=True)
        tracker.log(run, {f"progress_{a.split}{sfx}/conversations_without_swaps": float(np.mean([p["swap_windows"] == 0 for p in per_conv])),
                          f"progress_{a.split}{sfx}/swap_windows": sum(p["swap_windows"] for p in per_conv)}, k + 1)
    tp, fp, fn = (sum(p[x] for p in per_conv) for x in ("tp", "fp", "fn"))
    prec, rec = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    sw = sum(p["swap_windows"] for p in per_conv)
    out = {
        "act_precision": prec, "act_recall": rec, "act_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "bc_recall": sum(p["bc_hit"] for p in per_conv) / max(sum(p["bc_n"] for p in per_conv), 1),
        "bc_runs": sum(p["bc_n"] for p in per_conv),
        "bc_recall_any_slot": sum(p["bc_hit_any"] for p in per_conv) / max(sum(p["bc_n"] for p in per_conv), 1),
        "swap_windows": sw, "swap_windows_per_hour": sw / max(hours, 1e-9),
        "swap_windows_unflagged": sw - sum(p["swap_windows_flagged"] for p in per_conv),
        "swap_window_rate": sw / max(sum(p["two_speaker_windows"] for p in per_conv), 1),
        "conversations_without_swaps": float(np.mean([p["swap_windows"] == 0 for p in per_conv])),
        "uncertain_window_rate": sum(p["uncertain_windows"] for p in per_conv) / max(sum(p["windows"] for p in per_conv), 1),
        "conversations": len(per_conv), "hours": hours,
    }
    if a.real_clip:
        import soundfile as sf
        from scipy.signal import resample_poly

        y, rate = sf.read(a.real_clip, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(1)
        if rate != SR:
            g = np.gcd(rate, SR)
            y = resample_poly(y, SR // g, rate // g).astype(np.float32)
        y = y * np.float32(0.9 / max(float(np.abs(y).max()), 1e-6))
        windows = identity_pass(model, y, a.device, win_s=a.win_s, hop_s=a.hop_s)
        link(windows)
        p_act = memory_activity(windows, len(y)) > 0.5
        n = p_act.shape[1]
        e = np.log10(np.mean(np.pad(y, (0, n * 320 - len(y)))[: n * 320].reshape(n, 320) ** 2, 1) + 1e-12) * 10
        speech = e > e.max() - 40
        share = p_act[:, speech].mean(1)
        out.update({"real_dominant_share": float(share.max()), "real_other_share": float(share.min())})
    json.dump({"summary": out, "per_conversation": per_conv, "args": vars(a)}, open(a.out, "w"), indent=1)
    print("SUMMARY", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}), flush=True)
    tracker.log(run, {f"full_{a.split}{sfx}/{k}": v for k, v in out.items() if isinstance(v, (int, float))}, len(pairs))
    tracker.finish(run)


if __name__ == "__main__":
    main()
