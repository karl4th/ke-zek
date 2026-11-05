"""Train the two-block Kezek (M2S layout) on real Open Yap conversations.

    python experiments/K-M1/train_m.py --data /data/oy16 --split-file split.json \
        --init /data/init/kezek-k-t0c/kezek_tse.pt --flip-init --out run

Per 30 s span: block 1 gives activity of two slots and two slot embeddings from the mono span;
block 2 (KezekTSE, from K-T0c) separates K chunks of 4 s once per slot. One permutation of the
two real speakers onto the slots is chosen per span, over all of its losses at once:

    L(pi) = l_sep * mean_k,i state(y_ki, s_k,pi(i)) + l_mix * mix(k) + l_act * BCE(act, pi)
            + l_id * (1 - cos(z_i, ECAPA(clean source pi(i))))           + l_sep_slots * relu(cos(z_A, z_B) - 0.2)

ECAPA of the clean sources is a teacher for training only. The separator is conditioned on the
teacher embedding with probability p_tf (1 for the first 20% of steps, then down to 0 at 60%),
otherwise on block 1's own embedding; validation uses block 1 only (mono-only path).
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
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
import tracker  # noqa: E402
from m2s_model import IdentityEncoder, frames_20ms  # noqa: E402
from mdata import SpanSequences, collate  # noqa: E402
from rdata import split_ids, splits  # noqa: E402
from train_sep import state_items  # noqa: E402

from kezek import metrics  # noqa: E402
from kezek.losses import snr_tau  # noqa: E402
from kezek.model import KezekTSE, SpeakerEncoder  # noqa: E402

FRAME = 160


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def teacher_prob(step: int, total: int) -> float:
    a, b = 0.2 * total, 0.6 * total
    return 1.0 if step < a else max(0.0, 1.0 - (step - a) / (b - a))


def gather_chunks(x: torch.Tensor, starts: torch.Tensor, frames: int) -> torch.Tensor:
    """x [B, ..., T] sampled at 16 kHz, starts [B, K] in 10 ms frames -> [B, K, ..., frames*160]."""
    n = frames * FRAME
    out = [torch.stack([x[b, ..., s * FRAME : s * FRAME + n] for s in starts[b].tolist()])
           for b in range(x.shape[0])]
    return torch.stack(out)


def gather_frames(x: torch.Tensor, starts: torch.Tensor, frames: int) -> torch.Tensor:
    out = [torch.stack([x[b, ..., s : s + frames] for s in starts[b].tolist()])
           for b in range(x.shape[0])]
    return torch.stack(out)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = IdentityEncoder()
        self.sep = KezekTSE()


def forward_losses(model, se, b, dev, chunk_frames, p_tf, w, amp):
    mix, src = b["mix"].to(dev), b["src"].to(dev)
    act, disp, l_ref = b["act"].to(dev), b["disputed"].to(dev), b["l_ref"].to(dev)
    starts = b["chunks"].to(dev)
    B, K = starts.shape
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, z = model.enc(mix)
    logits, z = logits.float(), z.float()
    with torch.no_grad():
        teach = torch.stack([se(src[:, s]) for s in (0, 1)], dim=1)  # [B, 2, 192]
    nf = logits.shape[-1]
    act20 = frames_20ms(act, nf)
    perms = ((0, 1), (1, 0))
    bce = torch.stack([
        sum(F.binary_cross_entropy_with_logits(logits[:, i], act20[:, p[i]], reduction="none").mean(-1)
            for i in (0, 1)) / 2
        for p in perms
    ], dim=1)  # [B, 2]
    zn, tn = F.normalize(z, dim=-1), F.normalize(teach, dim=-1)
    ident = torch.stack([
        sum(1 - (zn[:, i] * tn[:, p[i]]).sum(-1) for i in (0, 1)) / 2 for p in perms
    ], dim=1)
    pick = (w["act"] * bce + w["id"] * ident).argmin(1)  # order used for teacher forcing
    use_teacher = torch.rand(B, device=dev) < p_tf
    teach_slots = torch.where(
        pick[:, None, None] == 0, teach, teach.flip(1)
    )  # slot i <- teacher of source pick(i)
    cond = torch.where(use_teacher[:, None, None], teach_slots, z)  # [B, 2, 192]

    mix_c = gather_chunks(mix, starts, chunk_frames)  # [B, K, n]
    src_c = gather_chunks(src, starts, chunk_frames)  # [B, K, 2, n]
    act_c = gather_frames(act, starts, chunk_frames)  # [B, K, 2, F]
    disp_c = gather_frames(disp, starts, chunk_frames)  # [B, K, F]
    n = mix_c.shape[-1]
    sep_in = mix_c[:, :, None].expand(B, K, 2, n).reshape(-1, n)
    sep_d = cond[:, None].expand(B, K, 2, cond.shape[-1]).reshape(-1, cond.shape[-1])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = model.sep(sep_in, sep_d)
    y = y.float().view(B, K, 2, n)

    sep = []
    for p in perms:
        terms = []
        for i in (0, 1):
            s = p[i]
            tgt = src_c[:, :, s].reshape(-1, n)
            a_s = act_c[:, :, s].reshape(-1, act_c.shape[-1])
            sil = (~act_c[:, :, s] & ~disp_c).reshape(-1, act_c.shape[-1])
            lr = l_ref[:, s, None].expand(B, K).reshape(-1)
            terms.append(state_items(y[:, :, i].reshape(-1, n), tgt, a_s, sil, lr).view(B, K))
        sep.append((terms[0] + terms[1]).mean(1) / 2)
    sep = torch.stack(sep, dim=1)  # [B, 2]
    mixc = -snr_tau(y.sum(2).reshape(-1, n), mix_c.reshape(-1, n), torch.ones_like(mix_c.reshape(-1, n)))
    mixc = mixc.view(B, K).mean(1)
    apart = F.relu((zn[:, 0] * zn[:, 1]).sum(-1) - 0.2)
    total = w["sep"] * sep + w["act"] * bce + w["id"] * ident  # [B, 2]
    best = total.argmin(1)
    loss = (total.gather(1, best[:, None])[:, 0] + w["mix"] * mixc + w["apart"] * apart).mean()
    rows = torch.arange(B, device=dev)
    parts = {
        "sep": float(sep[rows, best].mean()), "act_bce": float(bce[rows, best].mean()),
        "id": float(ident[rows, best].mean()), "mix": float(mixc.mean()), "apart": float(apart.mean()),
        "pick_agree": float((best == pick).float().mean()), "p_tf": p_tf,
    }
    return loss, parts, {"y": y, "best": best, "logits": logits, "act20": act20, "z": z, "teach": teach,
                         "mix_c": mix_c, "src_c": src_c, "act_c": act_c, "disp_c": disp_c}


@torch.no_grad()
def validate(model, se, dataset, dev, chunk_frames, amp, batch: int = 4) -> dict:
    """Mono-only path (p_tf = 0). Separation scored per slot with the span's one permutation;
    a chunk counts as swapped when the other permutation is better by > 3 dB summed SI-SDR."""
    model.eval()
    w = {"sep": 1.0, "act": 0.2, "id": 0.1, "mix": 0.1, "apart": 0.1}
    scores, f1n, f1d, swaps, swap_n, ids = [], 0.0, 0.0, 0, 0, []
    for i0 in range(0, len(dataset), batch):
        b = collate([dataset[i] for i in range(i0, min(i0 + batch, len(dataset)))])
        _, _, o = forward_losses(model, se, b, dev, chunk_frames, 0.0, w, amp)
        y, best = o["y"].cpu().numpy(), o["best"].cpu().numpy()
        src, mix = o["src_c"].cpu().numpy(), o["mix_c"].cpu().numpy()
        act, disp = o["act_c"].cpu().numpy(), o["disp_c"].cpu().numpy()
        l_ref = b["l_ref"].numpy()
        p = (torch.sigmoid(o["logits"]) > 0.5).float()
        for j in range(y.shape[0]):
            perm = (0, 1) if best[j] == 0 else (1, 0)
            pa = p[j] if best[j] == 0 else p[j].flip(0)
            tp = (pa * o["act20"][j]).sum()
            f1n += float(2 * tp)
            f1d += float(pa.sum() + o["act20"][j].sum())
            zt = F.normalize(o["z"][j], dim=-1)
            tt = F.normalize(o["teach"][j], dim=-1)
            ids.append(float((zt[0] * tt[perm[0]]).sum() + (zt[1] * tt[perm[1]]).sum()) / 2)
            for k in range(y.shape[1]):
                for i in (0, 1):
                    s = perm[i]
                    scores.append(metrics.segment_scores(
                        y[j, k, i], src[j, k, s], src[j, k, 1 - s], mix[j, k], act[j, k, s],
                        act[j, k, 1 - s], disp[j, k], float(l_ref[j, s])))
                if act[j, k, 0].sum() >= 50 and act[j, k, 1].sum() >= 50:
                    keep = metrics.si_sdr(y[j, k, 0], src[j, k, perm[0]]) + metrics.si_sdr(y[j, k, 1], src[j, k, perm[1]])
                    swap = metrics.si_sdr(y[j, k, 0], src[j, k, perm[1]]) + metrics.si_sdr(y[j, k, 1], src[j, k, perm[0]])
                    swap_n += 1
                    swaps += int(swap > keep + 3.0)
    model.train()
    out = metrics.aggregate(scores)
    out.update({"activity_f1": f1n / max(f1d, 1e-9), "id_cos": float(np.mean(ids)),
                "chunk_swap_rate": swaps / max(swap_n, 1), "chunk_swap_n": swap_n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--init", default="", help="KezekTSE state dict for block 2")
    ap.add_argument("--flip-init", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--bs", type=int, default=2, help="spans per batch (each has K chunks x 2 slots)")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--span-s", type=float, default=30.0)
    ap.add_argument("--chunk-s", type=float, default=4.0)
    ap.add_argument("--lr-enc", type=float, default=3e-4)
    ap.add_argument("--lr-sep", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--w-sep", type=float, default=1.0)
    ap.add_argument("--w-act", type=float, default=0.2, help="BCE is ~0.7 at start, sep ~10: see log")
    ap.add_argument("--w-id", type=float, default=0.1)
    ap.add_argument("--w-mix", type=float, default=0.1)
    ap.add_argument("--w-apart", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit", type=int, default=0, help="E0: train on this many fixed spans")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    train, val, test = splits(a.data, split_file=a.split_file or None)
    json.dump(split_ids((train, val, test)), open(f"{a.out}/split.json", "w"))
    json.dump({"args": vars(a)}, open(f"{a.out}/manifest.json", "w"), indent=1)
    w = {"sep": a.w_sep, "act": a.w_act, "id": a.w_id, "mix": a.w_mix, "apart": a.w_apart}
    chunk_frames = int(a.chunk_s * 100)

    ds_kw = dict(span_s=a.span_s, chunk_s=a.chunk_s, chunks=a.chunks)
    if a.overfit:
        fixed = SpanSequences(train, a.overfit, seed=a.seed, **ds_kw)
        train_set = torch.utils.data.Subset(fixed, [i % a.overfit for i in range(a.steps * a.bs)])
        val_set = fixed
    else:
        train_set = SpanSequences(train, a.steps * a.bs, seed=a.seed, **ds_kw)
        val_set = SpanSequences(val, a.n_val, seed=20_000, gain_db=0.0, peak=(0.9, 0.9), **ds_kw)
    ckpt_path = f"{a.out}/last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu") if a.resume and os.path.exists(ckpt_path) else None
    start = ckpt["step"] if ckpt else 0
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_set, range(start * a.bs, len(train_set))), batch_size=a.bs,
        num_workers=a.workers, collate_fn=collate, drop_last=True,
        prefetch_factor=4 if a.workers else None, persistent_workers=bool(a.workers), pin_memory=dev == "cuda",
    )
    se = SpeakerEncoder(device=dev)
    model = Model().to(dev)
    if a.init:
        model.sep.load_state_dict(torch.load(a.init, map_location="cpu"))
        if a.flip_init:
            with torch.no_grad():
                model.sep.dec.weight.neg_()
    print(f"encoder {sum(p.numel() for p in model.enc.parameters()) / 1e6:.2f}M, separator "
          f"{sum(p.numel() for p in model.sep.parameters()) / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": a.lr_enc},
        {"params": model.sep.parameters(), "lr": a.lr_sep},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, a.warmup, a.steps))
    log = []
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        log = ckpt["log"]

    name = os.path.basename(a.out.rstrip("/"))
    batches = iter(dl)
    run = tracker.init(name, {**vars(a), "weights": w}, ["two-block", "m2s-layout"], a.steps)
    for e in log:
        tracker.log(run, {f"val/{k}": v for k, v in e.items() if k != "step"}, e["step"])

    def run_eval(step):
        m = validate(model, se, val_set, dev, chunk_frames, a.amp)
        keep = ("overlap_sisdri", "target_solo_sisdr", "target_solo_level_db", "other_solo_leak_db",
                "backchannel_recall", "activity_f1", "id_cos", "chunk_swap_rate")
        print(f"EVAL step {step}: " + json.dumps({k: round(m[k], 3) for k in keep}), flush=True)
        log.append({"step": step, **m})
        json.dump(log, open(f"{a.out}/eval_log.json", "w"), indent=1)
        tracker.log(run, {f"val/{k}": v for k, v in m.items()}, step)
        return m

    if start == 0 and not a.overfit:
        run_eval(0)
    model.train()
    t0, last = time.time(), min(a.steps, a.stop_at or a.steps)
    for step, b in enumerate(batches, start + 1):
        p_tf = teacher_prob(step, a.steps)
        loss, parts, _ = forward_losses(model, se, b, dev, chunk_frames, p_tf, w, a.amp)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, batch skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        gn = {k: float(torch.nn.utils.clip_grad_norm_(getattr(model, k).parameters(), 5.0)) for k in ("enc", "sep")}
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 50 == 0:
            tracker.log(run, {"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()},
                              "train/gnorm_enc": gn["enc"], "train/gnorm_sep": gn["sep"]}, step)
        if step % 100 == 0 or step <= 3:
            el = time.time() - t0
            print(f"step {step:6d} loss {loss.item():8.3f} {json.dumps({k: round(v, 3) for k, v in parts.items()})} "
                  f"gn {gn['enc']:.2f}/{gn['sep']:.2f} {(step - start) / max(el, 1e-6):.2f} it/s", flush=True)
        if (step % a.eval_every == 0 and not a.overfit) or step == last:
            run_eval(step)
            torch.save({"enc": model.enc.state_dict(), "sep": model.sep.state_dict()}, f"{a.out}/model.pt")
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "log": log}, f"{ckpt_path}.tmp")
            os.replace(f"{ckpt_path}.tmp", ckpt_path)
        if step >= last:
            break
    tracker.finish(run)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
