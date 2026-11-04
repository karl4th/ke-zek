"""Long-form K-R1 evaluation: a whole excerpt of a real conversation and a voice sample of up to
30 s, as a user would run Kezek.

    python experiments/K-R1/evaluate_long.py --data /data/oy16 --method mixture --out m.json
    python experiments/K-R1/evaluate_long.py ... --method mono --model run/kezek_tse.pt
    python experiments/K-R1/evaluate_long.py ... --method block --model run/sep.pt [--no-conversation]

For every conversation of the split, one excerpt (``--excerpt-s``, fixed seed) and both people
as target in turn. The enrollment is the mixture on the target's longest solo stretches outside
the excerpt (2 s guard), up to ``--enroll-s``. Scores: the K-T1 segment metrics over the whole
excerpt, pooled, with 95% bootstrap intervals over excerpts. The test split needs ``--final``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block import assign, separate_long  # noqa: E402
from models import Extractor, SepTasNet  # noqa: E402
from rdata import _Conversation, _runs, disputed_frames, splits  # noqa: E402

from kezek import metrics  # noqa: E402
from kezek.model import SpeakerEncoder  # noqa: E402

SR, FRAME = 16_000, 160


def mono_long(model, mix, enr, d, device, win_s=4.0, hop_s=3.0, batch=16):
    win, hop = int(win_s * SR), int(hop_s * SR)
    n = len(mix)
    starts = list(range(0, max(n - win, 0), hop)) + [max(n - win, 0)]
    out, wsum = np.zeros(n, np.float32), np.zeros(n, np.float32)
    ov = win - hop
    e = torch.from_numpy(enr)[None].to(device)
    for i in range(0, len(starts), batch):
        ss = starts[i : i + batch]
        seg = torch.from_numpy(np.stack([mix[s : s + win] for s in ss])).to(device)
        with torch.no_grad():
            y = model(seg, e.expand(len(ss), -1), d.expand(len(ss), -1)).float().cpu().numpy()
        for s, yy in zip(ss, y, strict=True):
            w = np.ones(len(yy), np.float32)
            if s > 0:
                w[:ov] = np.linspace(0, 1, ov, dtype=np.float32)
            if s + win < n:
                w[-ov:] = np.linspace(1, 0, ov, dtype=np.float32)
            out[s : s + len(yy)] += yy * w
            wsum[s : s + len(yy)] += w
    return out / np.maximum(wsum, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", choices=("mixture", "mono", "block", "oracle_block"), required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--arch", default="kezek")
    ap.add_argument("--no-conversation", action="store_true", help="block: voice evidence only")
    ap.add_argument("--refine", type=int, default=5, help="block: centroid iterations (0 = off)")
    ap.add_argument("--rel-db", type=float, default=35.0, help="block: run detection level")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--excerpt-s", type=float, default=60.0)
    ap.add_argument("--enroll-s", type=float, default=20.0)
    ap.add_argument("--max-conversations", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-audio", default="", help="directory for a few example WAVs")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        raise SystemExit("the test split is read only with --final")
    _, val, test = splits(a.data)
    pairs = val if a.split == "val" else test
    if a.max_conversations:
        pairs = pairs[: a.max_conversations]
    dev = a.device
    se = SpeakerEncoder(device=dev)
    if a.method == "mono":
        model = Extractor(a.arch).to(dev).eval()
        model.net.load_state_dict(torch.load(a.model, map_location="cpu"))
    elif a.method == "block":
        model = SepTasNet().to(dev).eval()
        model.load_state_dict(torch.load(a.model, map_location="cpu"))
    rng = np.random.default_rng(a.seed)
    scores, infos, used = [], [], 0
    t0 = time.time()
    exc = int(a.excerpt_s * 100)
    for pair in pairs:
        c = _Conversation(pair)
        if c.frames < exc + 1000:
            continue
        start = int(rng.integers(0, c.frames - exc))
        for side in (0, 1):
            act_t, act_o = c.act[side], c.act[1 - side]
            mask = act_t & ~act_o
            mask[max(0, start - 200) : start + exc + 200] = False
            runs = _runs(mask)
            runs = runs[(runs[:, 1] - runs[:, 0]) >= 50]
            runs = runs[np.argsort(runs[:, 0] - runs[:, 1], kind="stable")]
            need, pieces = int(a.enroll_s * 100), []
            for r0, r1 in runs:
                take = min(int(r1 - r0), need - sum(q - p for p, q in pieces))
                pieces.append((int(r0), int(r0) + take))
                if sum(q - p for p, q in pieces) >= need:
                    break
            if sum(q - p for p, q in pieces) < min(need, 500):
                continue  # under 5 s of solo speech outside the excerpt
            pieces.sort()
            enr = np.concatenate([c.audio(0, p, q) + c.audio(1, p, q) for p, q in pieces])
            t = c.audio(side, start, start + exc)
            o = c.audio(1 - side, start, start + exc)
            mix = t + o
            g = 0.9 / max(float(np.abs(mix).max()), 1e-4)
            t, o, mix = t * g, o * g, mix * g
            enr = (enr * (0.9 / max(float(np.abs(enr).max()), 1e-4))).astype(np.float32)
            at, ao = act_t[start : start + exc], act_o[start : start + exc]
            with torch.no_grad():
                d = se(torch.from_numpy(enr)[None].to(dev))
            info = {}
            if a.method == "mixture":
                est = mix
            elif a.method == "mono":
                est = mono_long(model, mix, enr, d, dev)
            else:
                if a.method == "oracle_block":  # the clean tracks stand in for the separator
                    flip = np.random.default_rng([a.seed, used]).random() < 0.5
                    streams = np.stack([o, t]) if flip else np.stack([t, o])
                else:
                    streams = separate_long(model, mix, dev)
                est, info = assign(
                    streams, mix, d, se, dev, use_conversation=not a.no_conversation,
                    refine=a.refine, rel_db=a.rel_db,
                )
            m = np.repeat(at, FRAME)
            l_ref = float(np.mean(np.float64(t[m]) ** 2)) if m.any() else float(np.mean(t**2)) + 1e-12
            s = metrics.segment_scores(est, t, o, mix, at, ao, disputed_frames(at, ao), l_ref)
            s["conv"], s["side"] = pair.conversation_id, side
            scores.append(s)
            infos.append(info)
            if a.save_audio and used < 6:
                import soundfile as sf

                os.makedirs(a.save_audio, exist_ok=True)
                stem = f"{a.save_audio}/{pair.conversation_id}_{side}"
                sf.write(f"{stem}_mix.wav", mix, SR)
                sf.write(f"{stem}_target.wav", t, SR)
                sf.write(f"{stem}_est.wav", np.clip(est, -1, 1), SR)
            used += 1
        if used % 20 == 0:
            print(f"{used} excerpts, {time.time() - t0:.0f}s", flush=True)
    out = {
        "method": a.method, "model": a.model, "split": a.split, "excerpts": len(scores),
        "conversation_term": not a.no_conversation, "refine": a.refine, "rel_db": a.rel_db, "excerpt_s": a.excerpt_s,
        "enroll_s": a.enroll_s, **metrics.aggregate(scores),
    }
    out["ci"] = metrics.bootstrap_ci(scores, n_boot=500)
    if infos and infos[0]:
        out["block_info_mean"] = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump(scores, open(a.out.replace(".json", "_items.json"), "w"))
    keys = ("overlap_sisdri", "target_solo_sisdr", "target_solo_level_db", "other_solo_leak_db",
            "both_silent_db", "backchannel_recall", "backchannel_n")
    print(json.dumps({k: round(out[k], 3) for k in keys}), flush=True)


if __name__ == "__main__":
    main()
