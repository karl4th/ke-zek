"""K-R1: train KezekTSE on real two-person conversations (Open Yap, converted by convert.py).

    python experiments/K-R1/train.py --data /data/oy16 --init /data/init/kezek_tse.pt --out run
    python experiments/K-R1/train.py --data /data/oy16 --out run --lr 1e-3   # from scratch

The split is speaker-disjoint (seed 42) and saved to <out>/split.json; the test split is never
read here. Validation every --eval-every steps on two fixed sets of validation-speaker windows:
``nat`` (crops placed uniformly, the natural mix of conversation states) and ``ov`` (crops
around an overlap). The final checkpoint is kezek_tse.pt; best.pt (by nat/all/overlap_sisdr) is
kept for the report only.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-T1"))
from evaluate import validate  # noqa: E402
from rdata import RealWindows, collate, split_ids, splits  # noqa: E402

from kezek.losses import kt1_loss  # noqa: E402
from kezek.model import KezekTSE, SpeakerEncoder, si_sdr_loss  # noqa: E402


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def sisdr_only(est, clean, act_t):
    """Plain TSE objective: negative SI-SDR on windows where the target speaks >= 0.5 s."""
    ok = act_t.sum(-1) >= 50
    if not bool(ok.any()):
        return est.sum() * 0.0, {"neg_sisdr": 0.0}
    v = si_sdr_loss(est[ok], clean[ok]).mean()
    return v, {"neg_sisdr": float(v.detach())}


def git_commit() -> str:
    try:
        import subprocess

        c = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", HERE, "diff", "--quiet", "HEAD"]) != 0
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default="", help="K-T0c state dict; empty = random init")
    ap.add_argument("--loss", choices=("kt1", "sisdr"), default="kt1")
    ap.add_argument("--spk-weight", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--p-overlap", type=float, default=0.5)
    ap.add_argument("--sir-jitter-db", type=float, default=3.0)
    ap.add_argument("--max-train-conversations", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train, val, test = splits(a.data)
    ids = split_ids((train, val, test))
    if a.max_train_conversations:
        rng = np.random.default_rng(a.seed)
        train = [train[i] for i in sorted(rng.permutation(len(train))[: a.max_train_conversations])]
    manifest = {
        "commit": git_commit(),
        "args": vars(a),
        "split_counts": {k: len(v) for k, v in ids.items()},
        "train_used": len(train),
    }
    json.dump(ids, open(f"{a.out}/split.json", "w"))
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(json.dumps(manifest), flush=True)

    ckpt_path = f"{a.out}/last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu") if a.resume and os.path.exists(ckpt_path) else None
    start = ckpt["step"] if ckpt else 0

    train_set = RealWindows(
        train, a.steps * a.bs, seed=a.seed, p_overlap=a.p_overlap, sir_jitter_db=a.sir_jitter_db
    )
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_set, range(start * a.bs, len(train_set))),
        batch_size=a.bs,
        num_workers=a.workers,
        shuffle=False,
        drop_last=True,
        collate_fn=collate,
        prefetch_factor=4 if a.workers else None,
        pin_memory=True,
        persistent_workers=bool(a.workers),
    )
    val_sets = {
        "nat": RealWindows(val, a.n_val, seed=20_000, p_overlap=0.0, peak=(0.9, 0.9)),
        "ov": RealWindows(val, a.n_val, seed=30_000, p_overlap=1.0, peak=(0.9, 0.9)),
    }

    dev = a.device
    se = SpeakerEncoder(device=dev)
    model = KezekTSE().to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, a.warmup, a.steps))
    log = []
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        log = ckpt["log"]
        print(f"resumed from step {start}", flush=True)

    def run_eval(step: int):
        m = {}
        with torch.no_grad():
            for name, ds in val_sets.items():
                m.update({f"{name}/{k}": v for k, v in validate(model, se, ds, dev).items()})
        keep = (
            "all/overlap_sisdri", "all/overlap_sisdr", "all/target_solo_snr",
            "all/target_solo_level_db", "all/other_solo_leak_db", "all/both_silent_db",
            "all/backchannel_recall", "wrong_speaker", "pick_acc",
        )
        short = {f"{n}/{k}": round(m[f"{n}/{k}"], 3) for n in val_sets for k in keep}
        print(f"EVAL step {step}: {json.dumps(short)}", flush=True)
        log.append({"step": step, **m})
        json.dump(log, open(f"{a.out}/eval_log.json", "w"), indent=1)
        return m

    if start == 0:
        run_eval(0)
    model.train()
    t0 = time.time()
    last = min(a.steps, a.stop_at or a.steps)
    for step, b in enumerate(dl, start + 1):
        mix, clean = b["mix"].to(dev, non_blocking=True), b["clean"].to(dev, non_blocking=True)
        with torch.no_grad():
            d = se(b["enr_t"].to(dev, non_blocking=True))
        est = model(mix, d)
        if a.loss == "kt1":
            loss, parts = kt1_loss(
                est, clean, b["act_t"].to(dev), b["silent"].to(dev), b["l_ref"].to(dev), d, se,
                weights=(1.0, 0.3, a.spk_weight),
            )
        else:
            loss, parts = sisdr_only(est, clean, b["act_t"].to(dev))
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, batch skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 200 == 0 or step <= 5:
            el = time.time() - t0
            print(
                f"step {step:6d} loss {loss.item():8.3f} {parts} {el:6.0f}s "
                f"{(step - start) / max(el, 1e-6):.2f} it/s",
                flush=True,
            )
        if step % a.eval_every == 0 or step == last:
            m = run_eval(step)
            torch.save(model.state_dict(), f"{a.out}/kezek_tse.pt")
            score = m["nat/all/overlap_sisdr"]
            if np.isfinite(score) and score >= max(
                e["nat/all/overlap_sisdr"] for e in log if np.isfinite(e["nat/all/overlap_sisdr"])
            ):
                torch.save(model.state_dict(), f"{a.out}/best.pt")
            state = {
                "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "log": log,
            }
            torch.save(state, f"{ckpt_path}.tmp")
            os.replace(f"{ckpt_path}.tmp", ckpt_path)
            model.train()
        if step >= last:
            break
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
