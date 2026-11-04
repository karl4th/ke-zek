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
from models import ARCHS, Extractor  # noqa: E402
from rdata import RealWindows, collate, split_ids, splits  # noqa: E402
from validate import validate  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from kezek.losses import frames_to_samples, kt1_loss, residual_db, snr_tau  # noqa: E402
from kezek.model import SpeakerEncoder, si_sdr_loss  # noqa: E402


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


def masked_sisdr(est, ref, w, eps: float = 1e-8):
    """SI-SDR in dB of est against ref over the samples where w = 1, with a nonnegative
    scale. [B]"""
    e, r = est * w, ref * w
    alpha = (e * r).sum(-1, keepdim=True) / ((r * r).sum(-1, keepdim=True) + eps)
    alpha = alpha.clamp_min(0.0)  # an inverted output is an error, not a perfect estimate
    proj = alpha * r
    return 10 * torch.log10((proj**2).sum(-1) / (((e - proj) ** 2).sum(-1) + eps) + eps)


def state_loss(est, clean, act_t, silent, l_ref, level_weight: float, silence_weight: float):
    """Conversation-state loss that cannot be lowered by going quiet.

    * active target frames: -SI-SDR clamped to [-30, 30] dB (silence there scores -30, unlike
      an SNR whose floor is 0 dB) plus |level error| in dB, which fixes the output scale;
    * target-silent frames (edges excluded): the K-T1 penalty on the residual energy, relative
      to the target's speech level, above -30 dB.
    """
    n = est.shape[-1]
    wa, ws = frames_to_samples(act_t, n), frames_to_samples(silent, n)
    has_a = wa.sum(-1) >= 50 * 160
    has_s = ws.sum(-1) > 0
    si = masked_sisdr(est, clean, wa).clamp(-30.0, 30.0)
    e_out = (wa * est**2).sum(-1) + 1e-8
    e_ref = (wa * clean**2).sum(-1) + 1e-8
    level = (10 * torch.log10(e_out / e_ref)).abs()
    na = has_a.sum().clamp_min(1)
    l_act = -(si * has_a).sum() / na
    l_lvl = (level * has_a).sum() / na
    sil = F.relu(residual_db(est, clean, ws, l_ref) + 30.0)
    l_sil = (sil * has_s).sum() / has_s.sum().clamp_min(1)
    loss = l_act + level_weight * l_lvl + silence_weight * l_sil
    return loss, {
        "sisdr_active": float(-l_act.detach()),
        "level_db": float(l_lvl.detach()),
        "silence": float(l_sil.detach()),
    }


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
    ap.add_argument("--arch", choices=ARCHS, default="kezek")
    ap.add_argument("--init", default="", help="state dict of the network; empty = random init")
    ap.add_argument(
        "--flip-init",
        action="store_true",
        help="negate the KezekTSE decoder after loading: K-T0c outputs the target with inverted "
        "polarity (SI-SDR training cannot see the sign)",
    )
    ap.add_argument("--loss", choices=("state", "kt1", "sisdr"), default="state")
    ap.add_argument("--level-weight", type=float, default=0.5)
    ap.add_argument("--silence-weight", type=float, default=0.3)
    ap.add_argument("--spk-weight", type=float, default=0.1)
    ap.add_argument(
        "--dual",
        type=float,
        default=0.0,
        help="weight of the two-pass term: also extract the other speaker with their own "
        "enrollment, train that pass with the same loss, and ask the two outputs to sum to the "
        "mixture (0 = single pass)",
    )
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
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the extractor forward")
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
    model = Extractor(a.arch).to(dev)
    if a.init:
        model.net.load_state_dict(torch.load(a.init, map_location="cpu"))
        if a.flip_init:
            with torch.no_grad():
                model.net.dec.weight.neg_()
    print(f"{a.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
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
    def objective(est, ref, act, silent, l_ref, d):
        act, silent, l_ref = act.to(dev), silent.to(dev), l_ref.to(dev)
        if a.loss == "state":
            return state_loss(est, ref, act, silent, l_ref, a.level_weight, a.silence_weight)
        if a.loss == "kt1":
            return kt1_loss(
                est, ref, act, silent, l_ref, d, se, weights=(1.0, a.silence_weight, a.spk_weight)
            )
        return sisdr_only(est, ref, act)

    model.train()
    t0 = time.time()
    last = min(a.steps, a.stop_at or a.steps)
    for step, b in enumerate(dl, start + 1):
        mix, clean = b["mix"].to(dev, non_blocking=True), b["clean"].to(dev, non_blocking=True)
        enr = b["enr_t"].to(dev, non_blocking=True)
        with torch.no_grad():
            d = se(enr)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            est = model(mix, enr, d)
        est = est.float()
        loss, parts = objective(est, clean, b["act_t"], b["silent"], b["l_ref"], d)
        if a.dual:
            enr_o = b["enr_o"].to(dev, non_blocking=True)
            with torch.no_grad():
                d_o = se(enr_o)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                est_o = model(mix, enr_o, d_o)
            est_o = est_o.float()
            other = b["other"].to(dev)
            loss_o, _ = objective(est_o, other, b["act_o"], b["silent_o"], b["l_ref_o"], d_o)
            full = torch.ones_like(mix)
            consistency = -snr_tau(est + est_o, mix, full).mean()
            loss = 0.5 * (loss + loss_o) + a.dual * consistency
            parts["sum_snr"] = -float(consistency.detach())
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
            torch.save(model.net.state_dict(), f"{a.out}/kezek_tse.pt")
            score = m["nat/all/overlap_sisdr"]
            if np.isfinite(score) and score >= max(
                e["nat/all/overlap_sisdr"] for e in log if np.isfinite(e["nat/all/overlap_sisdr"])
            ):
                torch.save(model.net.state_dict(), f"{a.out}/best.pt")
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
