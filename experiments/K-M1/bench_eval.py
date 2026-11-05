"""kezek-v1-bench (HF dataset manifestro/kezek-v1-bench): real two-person podcasts, 10 English and
5 Kazakh, without reference tracks. Labels are pyannote turns, not ground truth.

    python experiments/K-M1/bench_eval.py --bench /data/bench --tse name=path ... \
        --s1 name=path ... --out bench.json

Target extractors (``--tse``, KezekTSE state dicts; ``name=path:flip`` negates the decoder):
* solo clips >= 20 s: the first 10 s are the voice sample, the rest must come back unchanged:
  SI-SDR(output, input), level, share of 0.1 s speech frames suppressed by > 15 dB;
* dialogs, with the voice sample from the same person's solo clip of the episode: on frames where
  only that person talks (pyannote), level vs the mixture and suppressed share ("kept"); on
  frames where only the other talks, output level vs the mixture ("leak", lower is better).
Stage 1 encoders (``--s1``, model.pt of train_m2s): on solo clips, share of speech frames in the
dominant slot and in the other; on dialogs, activity F1 against the pyannote turns with one
mapping per clip, and the share of turns whose dominant slot disagrees with that mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
import tracker  # noqa: E402

from kezek.infer import extract_long  # noqa: E402
from kezek.metrics import si_sdr  # noqa: E402
from kezek.model import KezekTSE, SpeakerEncoder  # noqa: E402

SR = 16_000
F01 = 1600  # 0.1 s


def load(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR and x.ndim == 1, path
    return x * np.float32(0.9 / max(float(np.abs(x).max()), 1e-6))


def energy(x: np.ndarray, f: int = F01) -> np.ndarray:
    n = len(x) // f
    return np.mean(np.float64(x[: n * f]).reshape(n, f) ** 2, 1) + 1e-12


def turn_mask(turns, who: str, other: str, n: int, f: int) -> np.ndarray:
    """Frames (of f samples) where only ``who`` talks, by the pyannote turns."""
    t = np.arange(n) * f / SR + f / SR / 2
    me = np.zeros(n, bool)
    ot = np.zeros(n, bool)
    for s, e, w in turns:
        sel = (t >= s) & (t < e)
        if w == who:
            me |= sel
        elif w == other:
            ot |= sel
    return me & ~ot


def tse_scores(models: dict, bench: str, episodes: list, dev: str):
    se = SpeakerEncoder(device=dev)

    def embed(x):
        with torch.no_grad():
            return se(torch.from_numpy(x)[None].to(dev))

    out = {}
    for name, m in models.items():
        solo, kept, leak = [], [], []
        for ep in episodes:
            singles = {c["speaker"]: c for c in ep["clips"] if c["kind"] == "single"}
            for c in singles.values():
                x = load(f"{bench}/{c['file']}")
                if len(x) < 20 * SR:
                    continue
                enr, test = x[: 10 * SR], x[10 * SR:]
                with torch.no_grad():
                    y = extract_long(m, test, embed(enr), win_s=4.0, overlap_s=1.0)
                ein, eout = energy(test), energy(y)
                sp = 10 * np.log10(ein) > 10 * np.log10(ein).max() - 40
                drop = 10 * np.log10(eout / ein)[sp]
                solo.append({"clip": c["file"], "sisdr": si_sdr(y, test),
                             "level_db": float(10 * np.log10(eout[sp].sum() / ein[sp].sum())),
                             "suppressed": float(np.mean(drop < -15))})
            for c in (c for c in ep["clips"] if c["kind"] == "dialog"):
                x = load(f"{bench}/{c['file']}")
                for who, other in (("A", "B"), ("B", "A")):
                    if who not in singles:
                        continue
                    d = embed(load(f"{bench}/{singles[who]['file']}"))
                    with torch.no_grad():
                        y = extract_long(m, x, d, win_s=4.0, overlap_s=1.0)
                    ein, eout = energy(x), energy(y)
                    me = turn_mask(c["turns"], who, other, len(ein), F01)
                    ot = turn_mask(c["turns"], other, who, len(ein), F01)
                    sp = 10 * np.log10(ein) > 10 * np.log10(ein).max() - 40
                    me, ot = me & sp, ot & sp
                    if me.sum() >= 10:
                        drop = 10 * np.log10(eout / ein)[me]
                        kept.append({"clip": c["file"], "who": who,
                                     "level_db": float(10 * np.log10(eout[me].sum() / ein[me].sum())),
                                     "suppressed": float(np.mean(drop < -15))})
                    if ot.sum() >= 10:
                        leak.append({"clip": c["file"], "who": who,
                                     "leak_db": float(10 * np.log10(eout[ot].sum() / ein[ot].sum()))})
        out[name] = {
            "solo_sisdr_db": float(np.mean([r["sisdr"] for r in solo])),
            "solo_level_db": float(np.mean([r["level_db"] for r in solo])),
            "solo_suppressed": float(np.mean([r["suppressed"] for r in solo])),
            "dialog_kept_level_db": float(np.mean([r["level_db"] for r in kept])),
            "dialog_kept_suppressed": float(np.mean([r["suppressed"] for r in kept])),
            "dialog_leak_db": float(np.mean([r["leak_db"] for r in leak])),
            "n_solo": len(solo), "n_kept": len(kept), "n_leak": len(leak),
            "per_clip": {"solo": solo, "kept": kept, "leak": leak},
        }
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                for k, v in out[name].items() if k != "per_clip"}), flush=True)
    return out


def s1_scores(models: dict, bench: str, episodes: list, dev: str, win_s: float, hop_s: float):
    from m2s_infer import build_memory, identity_pass, load as load_model, memory_activity

    out = {}
    for name, path in models.items():
        m = load_model(path, dev)
        solo, dia = [], []
        for ep in episodes:
            for c in ep["clips"]:
                x = load(f"{bench}/{c['file']}")
                w = identity_pass(m, x, dev, win_s=win_s, hop_s=hop_s)
                build_memory(w)
                p = memory_activity(w, len(x)) > 0.5
                n = p.shape[1]
                e = energy(np.pad(x, (0, max(0, n * 320 - len(x)))), 320)[:n]
                sp = 10 * np.log10(e) > 10 * np.log10(e).max() - 40
                if c["kind"] == "single":
                    share = p[:, sp].mean(1)
                    solo.append({"clip": c["file"], "dominant": float(share.max()), "other": float(share.min())})
                    continue
                lab = np.stack([turn_mask(c["turns"], s, "-", n, 320) for s in ("A", "B")])
                agree = [(p[0] & lab[0]).sum() + (p[1] & lab[1]).sum(), (p[0] & lab[1]).sum() + (p[1] & lab[0]).sum()]
                q = p if agree[0] >= agree[1] else p[::-1]
                tp, fp, fn = (q & lab).sum(), (q & ~lab).sum(), (~q & lab).sum()
                wrong = turns = 0
                for s, e_, who in c["turns"]:
                    i = 0 if who == "A" else 1
                    a, b = int(s * 50), max(int(s * 50) + 1, int(e_ * 50))
                    turns += 1
                    wrong += int(q[1 - i, a:b].mean() > q[i, a:b].mean())
                dia.append({"clip": c["file"], "tp": int(tp), "fp": int(fp), "fn": int(fn),
                            "wrong_turns": wrong, "turns": turns})
        tp, fp, fn = (sum(d[k] for d in dia) for k in ("tp", "fp", "fn"))
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[name] = {
            "solo_dominant_share": float(np.mean([r["dominant"] for r in solo])),
            "solo_other_share": float(np.mean([r["other"] for r in solo])),
            "dialog_act_f1": 2 * pr * rc / max(pr + rc, 1e-9),
            "dialog_wrong_turn_rate": sum(d["wrong_turns"] for d in dia) / max(sum(d["turns"] for d in dia), 1),
            "n_solo": len(solo), "n_dialog": len(dia), "per_clip": {"solo": solo, "dialog": dia},
        }
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                for k, v in out[name].items() if k != "per_clip"}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--tse", nargs="*", default=[], help="name=path[:flip]")
    ap.add_argument("--s1", nargs="*", default=[], help="name=path")
    ap.add_argument("--lang", nargs="*", default=["en", "kk"])
    ap.add_argument("--win-s", type=float, default=16.0)
    ap.add_argument("--hop-s", type=float, default=8.0)
    ap.add_argument("--revision", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    eps = [e for e in json.load(open(f"{a.bench}/segments.json")) if e["source"][:2] in a.lang]
    res = {}
    for lang in a.lang:
        sub = [e for e in eps if e["source"].startswith(lang)]
        models = {}
        for spec in a.tse:
            name, path = spec.split("=", 1)
            flip = path.endswith(":flip")
            m = KezekTSE().to(a.device).eval()
            m.load_state_dict(torch.load(path.removesuffix(":flip"), map_location="cpu"))
            if flip:
                with torch.no_grad():
                    m.dec.weight.neg_()
            models[name] = m
        if models:
            res[f"tse_{lang}"] = tse_scores(models, a.bench, sub, a.device)
        s1 = dict(spec.split("=", 1) for spec in a.s1)
        if s1:
            res[f"s1_{lang}"] = s1_scores(s1, a.bench, sub, a.device, a.win_s, a.hop_s)
    json.dump({"args": vars(a), "results": res}, open(a.out, "w"), indent=1)
    for group, by_model in res.items():
        for name, r in by_model.items():
            run = tracker.init(f"bench_{group}_{name}", {**vars(a), "model": name}, ["kezek-v1-bench", group], 1)
            tracker.log(run, {f"bench_{group}/{k}": v for k, v in r.items() if isinstance(v, (int, float))}, 0)
            tracker.finish(run)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
