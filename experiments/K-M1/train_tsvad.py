"""Second pass of stage 1: train ``ProfileHead`` (TS-VAD style) on a frozen stage 1 encoder.

The encoder of ``--init-model`` gives frame features of a 16 s span; the head gets them with the
two speakers' profiles (ECAPA of a reference clip of each person from elsewhere in the same
recording, through the same recording chain) and predicts the activity of each person in the
profiles' order: no permutation, so whose voice is where comes from the profiles, not from a
per-window guess. Activity labels as everywhere else (``KEZEK_LABELS`` / ``KEZEK_WORD_GAP``).

    python experiments/K-M1/train_tsvad.py --data /data/oy16 --split-file split.json \
        --init-model best.pt --aug-root /data/aug --out /data/runs/tsvad1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
import tracker  # noqa: E402
from m2s_data import SpanSequences, collate  # noqa: E402
from m2s_infer import load  # noqa: E402
from m2s_model import ProfileHead, frame_similarity, frames_20ms  # noqa: E402
from rdata import _runs, splits  # noqa: E402

SR = 16_000


@torch.no_grad()
def profiles(spk, ref: torch.Tensor, rng: np.random.Generator, lo_s: float, hi_s: float) -> torch.Tensor:
    """ref [B, 2, T] reference clips -> [B, 2, D] unit-norm ECAPA of a random crop of lo-hi s."""
    B, _, T = ref.shape
    rows, lens = [], []
    for b in range(B):
        for s in (0, 1):
            x = ref[b, s]
            nz = torch.nonzero(x.abs() > 1e-4).flatten()
            n = int(nz[-1]) + 1 if len(nz) else T
            L = min(n, int(rng.uniform(lo_s, hi_s) * SR))
            a = int(rng.integers(0, max(1, n - L + 1)))
            rows.append(x[a:a + L])
            lens.append(L)
    Lm = max(lens)
    batch = torch.stack([F.pad(r, (0, Lm - len(r))) for r in rows])
    rel = torch.tensor(lens, device=ref.device, dtype=torch.float32) / Lm
    with torch.autocast(ref.device.type, enabled=False):
        z = spk(batch.float(), rel)
    return F.normalize(z.float(), dim=-1).view(B, 2, -1)


def run(enc, spk, head, b, dev, rng, a, train=True):
    mix = b["mix"].to(dev, non_blocking=True)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=a.amp):
        h = enc.frames(mix, return_hidden=True)
    z = profiles(spk, b["ref"].to(dev), rng, *((a.ref_min, a.ref_max) if train else (a.ref_max, a.ref_max)))
    sim = frame_similarity(spk, mix, z, h.shape[1]) if a.sim else None
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=a.amp):
        logits = head(h.float(), z, sim)
    nf = logits.shape[-1]
    act = frames_20ms(b["act"].to(dev), nf)
    bc = frames_20ms(b["bc"].to(dev), nf)
    w = 1 + (a.bc_weight - 1) * bc
    loss = (F.binary_cross_entropy_with_logits(logits.float(), act, reduction="none") * w).sum() / w.sum()
    return loss, logits, act, bc


@torch.no_grad()
def validate(enc, spk, head, items, dev, a):
    head.eval()
    rng = np.random.default_rng(7)
    tp = fp = fn = fo = fo_n = 0.0
    bc_hit = bc_n = order_ok = n = 0
    for i in range(0, len(items), a.bs):
        b = collate(items[i:i + a.bs])
        _, logits, act, bc = run(enc, spk, head, b, dev, rng, a, train=False)
        p = (torch.sigmoid(logits) > 0.5).float()
        tp += float((p * act).sum())
        fp += float((p * (1 - act)).sum())
        fn += float(((1 - p) * act).sum())
        one = act.sum(1) == 1
        fo += float((p.sum(1) == 2)[one].sum())
        fo_n += float(one.sum())
        for j in range(p.shape[0]):
            keep = float((p[j, 0] * act[j, 0]).sum() + (p[j, 1] * act[j, 1]).sum())
            cross = float((p[j, 0] * act[j, 1]).sum() + (p[j, 1] * act[j, 0]).sum())
            if act[j].sum(-1).min() >= 25:
                n += 1
                order_ok += int(keep >= cross)
            pj, bj = p[j].cpu().numpy(), bc[j].bool().cpu().numpy()
            for s in (0, 1):
                for r0, r1 in _runs(bj[s]):
                    bc_n += 1
                    bc_hit += int(pj[s, r0:r1].mean() >= 0.5)
    head.train()
    prec, rec = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    return {"act_f1": 2 * prec * rec / max(prec + rec, 1e-9), "act_precision": prec, "act_recall": rec,
            "bc_recall": bc_hit / max(bc_n, 1), "false_overlap": fo / max(fo_n, 1), "order_accuracy": order_ok / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--init-model", required=True, help="stage 1 best.pt (frozen encoder)")
    ap.add_argument("--aug-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--sim", action="store_true", help="also per-frame cosine of 1.5 s segment embeddings with the profiles")
    ap.add_argument("--span-s", type=float, default=16.0)
    ap.add_argument("--single-p", type=float, default=0.25)
    ap.add_argument("--bc-weight", type=float, default=10.0)
    ap.add_argument("--ref-min", type=float, default=2.0, help="profile crop length range (s) in training")
    ap.add_argument("--ref-max", type=float, default=6.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    train, val, _ = splits(a.data, split_file=a.split_file or None)
    render = None
    if a.aug_root:
        from render import Render

        from kezek.augment import DomainBank

        r = a.aug_root
        render = Render(DomainBank.from_dirs(
            [f"{r}/RIRS_NOISES/simulated_rirs", f"{r}/RIRS_NOISES/real_rirs_isotropic_noises"],
            [f"{r}/musan/noise"], [f"{r}/musan/music"]))
    train_set = SpanSequences(train, a.steps * a.bs, seed=1, render=render, span_s=a.span_s, single_p=a.single_p)
    val_items = [SpanSequences(val, a.n_val, seed=20_000, gain_db=0.0, peak=(0.9, 0.9), render=render,
                               span_s=a.span_s)[i] for i in range(a.n_val)]
    m = load(a.init_model, dev)
    enc = m.enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    spk = enc._speaker[0]
    head = ProfileHead(enc.act.in_features, spk.dim, a.layers, a.sim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    start, best = 0, -1.0
    last = f"{a.out}/last.pt"
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu")
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["step"], ck["best"]
    cfg = {"init_model": a.init_model, "hdim": enc.act.in_features, "zdim": spk.dim, "layers": a.layers, "sim": a.sim,
           "labels": os.environ.get("KEZEK_LABELS", "energy"), "word_gap": os.environ.get("KEZEK_WORD_GAP", "0")}
    json.dump({"args": vars(a), "head": cfg}, open(f"{a.out}/manifest.json", "w"), indent=1)
    run_ = tracker.init(os.path.basename(a.out.rstrip("/")), {**vars(a), **cfg}, ["kezek-v3", "stage1", "tsvad"], a.steps)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, range(start * a.bs, len(train_set))),
                                     batch_size=a.bs, num_workers=a.workers, collate_fn=collate, drop_last=True,
                                     pin_memory=True, prefetch_factor=4 if a.workers else None,
                                     persistent_workers=bool(a.workers))

    def evaluate(step):
        nonlocal best
        v = validate(enc, spk, head, val_items, dev, a)
        print(f"EVAL step {step}: " + json.dumps({k: round(x, 3) for k, x in v.items()}), flush=True)
        tracker.log(run_, {f"val/{k}": x for k, x in v.items()}, step)
        score = v["act_f1"] + v["bc_recall"] + v["order_accuracy"]
        if score > best:
            best = score
            torch.save({"head": head.state_dict(), "cfg": cfg, "step": step, "val": v}, f"{a.out}/best.pt")

    if start == 0:
        evaluate(0)
    rng = np.random.default_rng(start)
    t0 = time.time()
    for step, b in enumerate(dl, start + 1):
        loss, *_ = run(enc, spk, head, b, dev, rng, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step:6d} loss {loss.item():.4f} {(step - start) / (time.time() - t0):.2f} it/s", flush=True)
            tracker.log(run_, {"train/loss": loss.item()}, step)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "cfg": cfg}, last)
        if step >= a.steps:
            break
    tracker.finish(run_)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
