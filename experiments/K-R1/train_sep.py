"""K-R1 block, stage 1: a two-speaker separator trained on real Open Yap conversations.

    python experiments/K-R1/train_sep.py --data /data/oy16 --out run \
        --init /data/init/kezek-k-t0c/kezek_tse.pt --flip-init

No enrollment: the separator puts each person on one output, whoever they are. The loss is the
K-R1 state loss for both outputs under the better of the two orders (permutation invariant), so
a silent person must come out silent, not as a copy of the other. Validation reports the target
metrics with the order chosen by the references (oracle assignment): the upper bound for the
assignment stage that follows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import BandSep, SepTasNet  # noqa: E402
from rdata import RealWindows, collate, split_ids, splits  # noqa: E402

from kezek import metrics  # noqa: E402
from kezek.losses import frames_to_samples, residual_db  # noqa: E402


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def state_items(est, ref, act, silent, l_ref, level_weight=0.5, silence_weight=0.3):
    """Per-window state loss [B]: -SI-SDR (nonnegative scale, clamped to +-30 dB) and |level
    error| on active frames, residual penalty on silent frames."""
    n = est.shape[-1]
    wa, ws = frames_to_samples(act, n), frames_to_samples(silent, n)
    has_a = (wa.sum(-1) >= 50 * 160).float()
    has_s = (ws.sum(-1) > 0).float()
    e, r = est * wa, ref * wa
    alpha = ((e * r).sum(-1, keepdim=True) / ((r * r).sum(-1, keepdim=True) + 1e-8)).clamp_min(0)
    proj = alpha * r
    si = 10 * torch.log10((proj**2).sum(-1) / (((e - proj) ** 2).sum(-1) + 1e-8) + 1e-8)
    si = si.clamp(-30.0, 30.0)
    level = (10 * torch.log10(((e**2).sum(-1) + 1e-8) / ((r**2).sum(-1) + 1e-8))).abs()
    sil = F.relu(residual_db(est, ref, ws, l_ref) + 30.0)
    return has_a * (-si + level_weight * level) + has_s * silence_weight * sil


def pit(y, b, dev):
    """Loss per window under the better order, and that order (0: y0 is the target)."""
    t, o = b["clean"].to(dev), b["other"].to(dev)
    at, ao = b["act_t"].to(dev), b["act_o"].to(dev)
    st, so = b["silent"].to(dev), b["silent_o"].to(dev)
    lt, lo = b["l_ref"].to(dev), b["l_ref_o"].to(dev)
    p0 = state_items(y[:, 0], t, at, st, lt) + state_items(y[:, 1], o, ao, so, lo)
    p1 = state_items(y[:, 1], t, at, st, lt) + state_items(y[:, 0], o, ao, so, lo)
    best = torch.minimum(p0, p1)
    return 0.5 * best, (p1 < p0).long()


@torch.no_grad()
def validate(model, dataset, dev, batch: int = 16, amp: bool = False) -> dict:
    model.eval()
    scores = []
    for i0 in range(0, len(dataset), batch):
        items = [dataset[i] for i in range(i0, min(i0 + batch, len(dataset)))]
        b = {k: v for k, v in collate(items).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = model(b["mix"].to(dev))
        y = y.float()
        _, order = pit(y, b, dev)
        est = y[torch.arange(len(items)), order].cpu().numpy()
        for k, it in enumerate(items):
            scores.append(
                metrics.segment_scores(
                    est[k], it["clean"], it["other"], it["mix"], it["act_t"], it["act_o"],
                    it["disputed"], float(it["l_ref"]),
                )
            )
    model.train()
    return metrics.aggregate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=("tasnet", "band"), default="tasnet")
    ap.add_argument("--init", default="")
    ap.add_argument("--flip-init", action="store_true")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--crop-s", type=float, default=4.0, help="training window, seconds")
    ap.add_argument("--init-sep", default="", help="start from a separator state dict (sep.pt)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train, val, test = splits(a.data)
    json.dump(split_ids((train, val, test)), open(f"{a.out}/split.json", "w"))
    json.dump({"args": vars(a)}, open(f"{a.out}/manifest.json", "w"), indent=1)

    ckpt_path = f"{a.out}/last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu") if a.resume and os.path.exists(ckpt_path) else None
    start = ckpt["step"] if ckpt else 0
    ds = RealWindows(
        train, a.steps * a.bs, seed=a.seed, p_overlap=0.5, sir_jitter_db=3.0, crop_s=a.crop_s
    )
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, range(start * a.bs, len(ds))), batch_size=a.bs,
        num_workers=a.workers, collate_fn=collate, drop_last=True, prefetch_factor=4,
        pin_memory=True, persistent_workers=True,
    )
    val_sets = {
        "nat": RealWindows(val, a.n_val, seed=20_000, p_overlap=0.0, peak=(0.9, 0.9)),
        "ov": RealWindows(val, a.n_val, seed=30_000, p_overlap=1.0, peak=(0.9, 0.9)),
    }
    model = (SepTasNet() if a.arch == "tasnet" else BandSep()).to(dev)
    print(f"{a.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    if a.init and a.arch == "tasnet":
        model.net.load_state_dict(torch.load(a.init, map_location="cpu"))
        with torch.no_grad():
            if a.flip_init:
                model.net.dec.weight.neg_()
            model.init_masks_from_net()
    if a.init_sep:
        model.load_state_dict(torch.load(a.init_sep, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, a.warmup, a.steps))
    log = []
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        log = ckpt["log"]

    def run_eval(step):
        m = {f"{n}/{k}": v for n, d in val_sets.items() for k, v in validate(model, d, dev, amp=a.amp).items()}
        keep = ("overlap_sisdri", "target_solo_sisdr", "target_solo_level_db", "other_solo_leak_db",
                "backchannel_recall")
        print(f"EVAL step {step}: " + json.dumps({f"{n}/{k}": round(m[f"{n}/{k}"], 3) for n in val_sets for k in keep}), flush=True)
        log.append({"step": step, **m})
        json.dump(log, open(f"{a.out}/eval_log.json", "w"), indent=1)

    if start == 0:
        run_eval(0)
    t0, last = time.time(), min(a.steps, a.stop_at or a.steps)
    for step, b in enumerate(dl, start + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            y = model(b["mix"].to(dev, non_blocking=True))
        loss = pit(y.float(), b, dev)[0].mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 200 == 0 or step <= 3:
            el = time.time() - t0
            print(f"step {step:6d} loss {loss.item():8.3f} {(step - start) / max(el, 1e-6):.2f} it/s", flush=True)
        if step % a.eval_every == 0 or step == last:
            run_eval(step)
            torch.save(model.state_dict(), f"{a.out}/sep.pt")
            score = log[-1]["nat/overlap_sisdri"]
            if np.isfinite(score) and score >= max(e["nat/overlap_sisdri"] for e in log):
                torch.save(model.state_dict(), f"{a.out}/best.pt")
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "log": log}, f"{ckpt_path}.tmp")
            os.replace(f"{ckpt_path}.tmp", ckpt_path)
        if step >= last:
            break
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
