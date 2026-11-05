"""Train the two-block Kezek in the M2S v0.1 stages.

    --stage enc   block 1 only: activity BCE + identity InfoNCE (clean references as teacher)
    --stage sep   block 2 only, encoder frozen: separator conditioned on clean-reference
                  embeddings in speaker order (teacher forcing); the order is known, so no
                  permutation is chosen (with one, the separator learned to ignore the condition)
    --stage e2e   both blocks: separator conditioned on block 1's mono-only embeddings, teacher
                  forcing decreasing from 1 (first 20% of steps) to 0 (at 60%)
    --calibrate N measure each objective's gradient norm over N batches and write the weights
                  that give the four objectives equal gradient scale (calibration.json)

One permutation of the two real speakers onto the slots is chosen per sequence (30 s span with
K adjacent 4 s chunks), from the summed weighted objectives; validation uses the mono-only path
and scores every span with one fixed permutation (never a minimum per chunk).
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
from m2s_data import SpanSequences, collate  # noqa: E402
from m2s_losses import (  # noqa: E402
    info_nce, masked_mean, mixture_l1, mr_stft_l1, neg_sisdr_active, silence_energy,
)
from m2s_model import ConditionedSeparator, IdentityEncoder, frames_20ms  # noqa: E402
from rdata import split_ids, splits  # noqa: E402

from kezek import metrics  # noqa: E402

FRAME = 160
TERMS = ("sep", "act", "id", "mix")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = IdentityEncoder()
        self.sep = ConditionedSeparator()


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def teacher_prob(step: int, total: int) -> float:
    a, b = 0.2 * total, 0.6 * total
    return 1.0 if step < a else max(0.0, 1.0 - (step - a) / (b - a))


def chunk_view(x, starts, k, frames):
    """x [B, ..., T] -> [B, k, ..., frames*160] for k adjacent chunks from starts [B] (frames)."""
    n = frames * FRAME
    return torch.stack([
        torch.stack([x[b, ..., (s + j * frames) * FRAME:(s + j * frames) * FRAME + n] for j in range(k)])
        for b, s in enumerate(starts.tolist())
    ])


def chunk_frames(x, starts, k, frames):
    return torch.stack([
        torch.stack([x[b, ..., s + j * frames:s + (j + 1) * frames] for j in range(k)])
        for b, s in enumerate(starts.tolist())
    ])


def objectives(model, b, dev, stage, p_tf, k, frames, amp):
    """Raw objectives per sequence and permutation, plus tensors for validation."""
    mix, src, act = b["mix"].to(dev), b["src"].to(dev), b["act"].to(dev)
    disp, l_ref, ref = b["disputed"].to(dev), b["l_ref"].to(dev), b["ref"].to(dev)
    starts = b["seq_start"]
    B = mix.shape[0]
    enc_grad = stage in ("enc", "e2e")
    with torch.set_grad_enabled(enc_grad and torch.is_grad_enabled()):
        logits, z = model.enc(mix)
        r = model.enc.embed_single(ref.flatten(0, 1)).view(B, 2, -1)
    act20 = frames_20ms(act, logits.shape[-1])
    bce = torch.stack([
        (F.binary_cross_entropy_with_logits(logits[:, 0], act20[:, p[0]], reduction="none").mean(-1)
         + F.binary_cross_entropy_with_logits(logits[:, 1], act20[:, p[1]], reduction="none").mean(-1)) / 2
        for p in ((0, 1), (1, 0))
    ], dim=1)
    nce = info_nce(z, r)
    raw = {"bce": bce, "nce": nce}
    out = {"logits": logits, "act20": act20, "z": z, "r": r}
    if stage == "enc":
        return raw, out

    mix_c = chunk_view(mix, starts, k, frames)  # [B, k, n]
    src_c = chunk_view(src, starts, k, frames)  # [B, k, 2, n]
    act_c = chunk_frames(act, starts, k, frames)  # [B, k, 2, F]
    disp_c = chunk_frames(disp, starts, k, frames)  # [B, k, F]
    n = mix_c.shape[-1]
    if stage == "sep":
        cond = r.detach()  # slot i <- clean reference of speaker i
    else:
        pick = (bce + nce).argmin(1)
        teach = torch.where(pick[:, None, None] == 0, r, r.flip(1)).detach()
        use = (torch.rand(B, device=dev) < p_tf)[:, None, None]
        cond = torch.where(use, teach, z)
    sep_in = mix_c[:, :, None].expand(B, k, 2, n).reshape(-1, n)
    sep_z = cond[:, None].expand(B, k, 2, cond.shape[-1]).reshape(-1, cond.shape[-1])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = model.sep(sep_in, sep_z)
    y = y.float().view(B, k, 2, n)

    si, stft, sil, o2s = [], [], [], []
    solo = act_c & ~act_c.flip(2)  # [B, k, 2, F]
    emb_y = model.enc.embed_single(y.reshape(-1, n)).view(B, k, 2, -1)
    for p in ((0, 1), (1, 0)):
        t_si, t_stft, t_sil, t_o2s = [], [], [], []
        for i in (0, 1):
            s = p[i]
            yi, si_ = y[:, :, i].reshape(-1, n), src_c[:, :, s].reshape(-1, n)
            a_s = act_c[:, :, s].reshape(-1, act_c.shape[-1])
            v, valid = neg_sisdr_active(yi, si_, a_s)
            t_si.append(masked_mean(v.view(B, k), valid.view(B, k)))
            t_stft.append(mr_stft_l1(yi, si_, mix_c.reshape(-1, n)).view(B, k).mean(1))
            pen, pv = silence_energy(yi, a_s, disp_c.reshape(-1, disp_c.shape[-1]),
                                     l_ref[:, s, None].expand(B, k).reshape(-1))
            t_sil.append(masked_mean(pen.view(B, k), pv.view(B, k)))
            cos = (emb_y[:, :, i] * cond[:, None, i]).sum(-1)  # output vs its own slot
            t_o2s.append(masked_mean(1 - cos, solo[:, :, s].sum(-1) >= 100))
        si.append((t_si[0] + t_si[1]) / 2)
        stft.append((t_stft[0] + t_stft[1]) / 2)
        sil.append((t_sil[0] + t_sil[1]) / 2)
        o2s.append((t_o2s[0] + t_o2s[1]) / 2)
    raw.update({
        "sisdr": torch.stack(si, 1), "stft": torch.stack(stft, 1),
        "silence": torch.stack(sil, 1), "out2slot": torch.stack(o2s, 1),
    })
    any_act = (act_c[:, :, 0] | act_c[:, :, 1]).reshape(-1, act_c.shape[-1])
    raw["mix"] = mixture_l1(mix_c.reshape(-1, n), y.sum(2).reshape(-1, n), any_act).view(B, k).mean(1)
    out.update({"y": y, "mix_c": mix_c, "src_c": src_c, "act_c": act_c, "disp_c": disp_c})
    return raw, out


def combine(raw, w, stage):
    """Per-permutation objectives -> L_sep, L_act, L_id ([B, 2]) and L_mix ([B])."""
    terms = {"act": raw["bce"], "id": raw["nce"]}
    if stage != "enc":
        terms["sep"] = raw["sisdr"] + raw["stft"]
        terms["act"] = raw["bce"] + raw["silence"]
        terms["id"] = raw["nce"] + raw["out2slot"]
    total = sum(w[t] * v for t, v in terms.items())
    if stage == "sep":
        # the slots are the clean references in speaker order: no permutation to choose, so the
        # separator must follow its conditioning (a free permutation lets it ignore it)
        best = torch.zeros(total.shape[0], dtype=torch.long, device=total.device)
    else:
        best = total.argmin(1)
    rows = torch.arange(total.shape[0], device=total.device)
    loss = total[rows, best]
    if stage != "enc":
        loss = loss + w["mix"] * raw["mix"]
        terms["mix"] = torch.stack([raw["mix"], raw["mix"]], 1)
    picked = {t: v[rows, best] for t, v in terms.items()}
    return loss.mean(), picked, best


@torch.no_grad()
def validate(model, dataset, dev, stage, k, frames, amp, batch=4):
    model.eval()
    scores, si_all, si_ov, si_solo = [], [], [], []
    tp = fp = fn = 0.0
    swaps = swap_n = 0
    for i0 in range(0, len(dataset), batch):
        b = collate([dataset[i] for i in range(i0, min(i0 + batch, len(dataset)))])
        raw, o = objectives(model, b, dev, stage if stage != "e2e" else "e2e", 0.0, k, frames, amp)
        # one permutation per span. In the separator stage the slots are the clean references in
        # speaker order; otherwise block 1's mono path decides (activity + identity agreement).
        if stage == "sep":
            perm_idx = np.zeros(len(b["conv"]), dtype=int)
        else:
            perm_idx = (raw["bce"] + raw["nce"]).argmin(1).cpu().numpy()
        p_act = (torch.sigmoid(o["logits"]) > 0.5).float()
        for j in range(len(perm_idx)):
            perm = (0, 1) if perm_idx[j] == 0 else (1, 0)
            pa = p_act[j]
            lab = o["act20"][j][list(perm)]  # slot i is scored against speaker perm[i]
            tp += float((pa * lab).sum())
            fp += float((pa * (1 - lab)).sum())
            fn += float(((1 - pa) * lab).sum())
            if stage == "enc":
                continue
            y, src, mix = o["y"][j].cpu().numpy(), o["src_c"][j].cpu().numpy(), o["mix_c"][j].cpu().numpy()
            act, disp = o["act_c"][j].cpu().numpy(), o["disp_c"][j].cpu().numpy()
            l_ref = b["l_ref"][j].numpy()
            for c in range(y.shape[0]):
                for i in (0, 1):
                    s = perm[i]
                    sc = metrics.segment_scores(y[c, i], src[c, s], src[c, 1 - s], mix[c], act[c, s],
                                                act[c, 1 - s], disp[c], float(l_ref[s]))
                    scores.append(sc)
                    if act[c, s].sum() >= 50:
                        gain = metrics.si_sdr(y[c, i], src[c, s]) - metrics.si_sdr(mix[c], src[c, s])
                        si_all.append(gain)
                        (si_ov if (act[c, 1 - s].sum() >= 50) else si_solo).append(gain)
                if act[c, 0].sum() >= 50 and act[c, 1].sum() >= 50:
                    keep = metrics.si_sdr(y[c, 0], src[c, perm[0]]) + metrics.si_sdr(y[c, 1], src[c, perm[1]])
                    other = metrics.si_sdr(y[c, 0], src[c, perm[1]]) + metrics.si_sdr(y[c, 1], src[c, perm[0]])
                    swap_n += 1
                    swaps += int(other > keep + 3.0)
    model.train()
    out = {"act_precision": tp / max(tp + fp, 1e-9), "act_recall": tp / max(tp + fn, 1e-9)}
    out["act_f1"] = 2 * out["act_precision"] * out["act_recall"] / max(out["act_precision"] + out["act_recall"], 1e-9)
    if stage != "enc":
        out.update(metrics.aggregate(scores))
        out.update({
            "sisdri_overall": float(np.mean(si_all)) if si_all else float("nan"),
            "sisdri_overlap_windows": float(np.mean(si_ov)) if si_ov else float("nan"),
            "sisdri_single_windows": float(np.mean(si_solo)) if si_solo else float("nan"),
            "chunk_swap_rate": swaps / max(swap_n, 1), "chunk_swap_n": swap_n,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("enc", "sep", "e2e"), required=True)
    ap.add_argument("--init-model", default="", help="model.pt of an earlier stage")
    ap.add_argument("--init-tse", default="", help="KezekTSE state dict for block 2 (K-T0c)")
    ap.add_argument("--flip-init", action="store_true", help="K-T0c outputs inverted polarity")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8, help="sequences per batch")
    ap.add_argument("--chunks", type=int, default=2, help="adjacent 4 s chunks per sequence")
    ap.add_argument("--lr-enc", type=float, default=3e-4)
    ap.add_argument("--lr-sep", type=float, default=3e-4)
    ap.add_argument("--weights", default="", help="calibration.json; default 1/0.2/0.1/0.1")
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="validate --init-model and exit")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit", type=int, default=0, help="E0: train on this many fixed sequences")
    ap.add_argument("--train-hours", type=float, default=0, help="E1 2-5 h, E2 50 h, E3 100 h; 0 = all")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    train, val, test = splits(a.data, split_file=a.split_file or None)
    if a.train_hours:
        order = np.random.default_rng(a.seed).permutation(len(train))
        chosen, hours = [], 0.0
        for i in order:
            if hours >= a.train_hours:
                break
            chosen.append(train[i])
            hours += train[i].frames / 16_000 / 3600
        train = sorted(chosen, key=lambda p: p.conversation_id)
        print(f"train subset: {len(train)} conversations, {hours:.1f} h", flush=True)
    json.dump(split_ids((train, val, test)), open(f"{a.out}/split.json", "w"))
    w = {"sep": 1.0, "act": 0.2, "id": 0.1, "mix": 0.1}
    if a.weights:
        w.update(json.load(open(a.weights))["weights"])
    json.dump({"args": vars(a), "weights": w}, open(f"{a.out}/manifest.json", "w"), indent=1)
    frames = 400

    if a.overfit:
        fixed = SpanSequences(train, a.overfit, seed=a.seed, chunks=a.chunks)
        items = [fixed[i] for i in range(a.overfit)]

        class Fixed(torch.utils.data.Dataset):
            def __len__(self):
                return a.steps * a.bs

            def __getitem__(self, i):
                return items[i % a.overfit]
        train_set, val_set = Fixed(), fixed
    else:
        train_set = SpanSequences(train, a.steps * a.bs, seed=a.seed, chunks=a.chunks)
        val_set = SpanSequences(val, a.n_val, seed=20_000, chunks=a.chunks, gain_db=0.0, peak=(0.9, 0.9))
    ckpt_path = f"{a.out}/last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu") if a.resume and os.path.exists(ckpt_path) else None
    start = ckpt["step"] if ckpt else 0
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_set, range(start * a.bs, len(train_set))), batch_size=a.bs,
        num_workers=a.workers, collate_fn=collate, drop_last=True, pin_memory=True,
        prefetch_factor=4 if a.workers else None, persistent_workers=bool(a.workers),
    )
    model = Model().to(dev)
    if a.init_tse:
        model.sep.tse.load_state_dict(torch.load(a.init_tse, map_location="cpu"))
        if a.flip_init:
            with torch.no_grad():
                model.sep.tse.dec.weight.neg_()
    if a.init_model:
        state = torch.load(a.init_model, map_location="cpu")
        model.enc.load_state_dict(state["enc"])
        if "sep" in state and a.stage == "e2e":
            model.sep.load_state_dict(state["sep"])
    if a.stage == "sep":
        for p in model.enc.parameters():
            p.requires_grad_(False)
    groups = []
    if a.stage in ("enc", "e2e"):
        groups.append({"params": model.enc.parameters(), "lr": a.lr_enc})
    if a.stage in ("sep", "e2e"):
        groups.append({"params": model.sep.parameters(), "lr": a.lr_sep})
    n_enc = sum(p.numel() for p in model.enc.parameters())
    n_sep = sum(p.numel() for p in model.sep.parameters())
    print(f"stage {a.stage}: encoder {n_enc / 1e6:.2f}M, separator {n_sep / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    warmup = max(1, int(0.02 * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, a.steps))
    log, best_score = [], -1e9
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        log, best_score = ckpt["log"], ckpt.get("best_score", -1e9)

    if a.eval_only:
        if a.init_model:
            state = torch.load(a.init_model, map_location="cpu")
            model.sep.load_state_dict(state["sep"])
        m = validate(model, val_set, dev, a.stage, a.chunks, frames, a.amp)
        json.dump(m, open(f"{a.out}/eval_only.json", "w"), indent=1)
        print("EVAL ONLY", json.dumps({kk: round(v, 3) for kk, v in m.items() if isinstance(v, float)}), flush=True)
        return

    if a.calibrate:
        # gradient norm of each unit-weight objective over the trainable parameters
        model.train()
        sums = {t: [] for t in TERMS}
        params = [p for g in groups for p in g["params"]]
        for step, b in enumerate(dl, 1):
            raw, _ = objectives(model, b, dev, a.stage, 1.0, a.chunks, frames, a.amp)
            unit = {t: 1.0 for t in TERMS}
            _, picked, _ = combine(raw, unit, a.stage)
            for t, v in picked.items():
                g = torch.autograd.grad(v.mean(), params, retain_graph=True, allow_unused=True)
                sums[t].append(float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None))))
            if step >= a.calibrate:
                break
        norms = {t: float(np.mean(v)) for t, v in sums.items() if v}
        ref = norms.get("sep", norms.get("act"))
        weights = {t: ref / max(n, 1e-12) for t, n in norms.items()}
        json.dump({"stage": a.stage, "batches": a.calibrate, "grad_norm_unit_weight": norms,
                   "weights": weights}, open(f"{a.out}/calibration.json", "w"), indent=1)
        print("CALIBRATION", json.dumps({"grad_norms": norms, "weights": weights}), flush=True)
        return

    name = os.path.basename(a.out.rstrip("/"))
    batches = iter(dl)
    run = tracker.init(name, {**vars(a), "weights": w, "params_enc": n_enc, "params_sep": n_sep},
                       ["two-block", "m2s-v0.1", a.stage], a.steps)
    for e in log:
        tracker.log(run, {f"val/{kk}": v for kk, v in e.items() if kk != "step"}, e["step"])

    def run_eval(step):
        nonlocal best_score
        m = validate(model, val_set, dev, a.stage, a.chunks, frames, a.amp)
        show = ("act_f1", "sisdri_overall", "sisdri_overlap_windows", "target_solo_sisdr", "target_solo_level_db",
                "other_solo_leak_db", "both_silent_db", "backchannel_recall", "chunk_swap_rate")
        print(f"EVAL step {step}: " + json.dumps({kk: round(m[kk], 3) for kk in show if kk in m}), flush=True)
        log.append({"step": step, **m})
        json.dump(log, open(f"{a.out}/eval_log.json", "w"), indent=1)
        tracker.log(run, {f"val/{kk}": v for kk, v in m.items()}, step)
        score = m.get("sisdri_overall", m["act_f1"])
        if np.isfinite(score) and score > best_score:
            best_score = score
            torch.save({"enc": model.enc.state_dict(), "sep": model.sep.state_dict()}, f"{a.out}/best.pt")

    if start == 0 and not a.overfit:
        run_eval(0)
    model.train()
    t0, last = time.time(), min(a.steps, a.stop_at or a.steps)
    for step, b in enumerate(batches, start + 1):
        p_tf = teacher_prob(step, a.steps) if a.stage == "e2e" else 1.0
        raw, _ = objectives(model, b, dev, a.stage, p_tf, a.chunks, frames, a.amp)
        loss, picked, _ = combine(raw, w, a.stage)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, batch skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 25 == 0:
            vals = {"train/loss": loss.item(), "train/grad_norm": gn, "train/p_tf": p_tf}
            vals.update({f"train/raw_{t}": float(v.mean()) for t, v in picked.items()})
            vals.update({f"train/weighted_{t}": w[t] * float(v.mean()) for t, v in picked.items()})
            tracker.log(run, vals, step)
        if step % 100 == 0 or step <= 3:
            el = time.time() - t0
            print(f"step {step:6d} loss {loss.item():8.3f} "
                  + json.dumps({t: round(float(v.mean()), 3) for t, v in picked.items()})
                  + f" gn {gn:.2f} p_tf {p_tf:.2f} {(step - start) / max(el, 1e-6):.2f} it/s", flush=True)
        if step % 500 == 0 or step == last:
            if (step % a.eval_every == 0 and not a.overfit) or step == last:
                run_eval(step)
            torch.save({"enc": model.enc.state_dict(), "sep": model.sep.state_dict()}, f"{a.out}/model.pt")
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "log": log, "best_score": best_score}, f"{ckpt_path}.tmp")
            os.replace(f"{ckpt_path}.tmp", ckpt_path)
        if step >= last:
            break
    tracker.finish(run)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
