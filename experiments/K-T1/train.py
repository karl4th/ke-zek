"""K-T1: fine-tune the K-T0c checkpoint on simulated conversations with domain augmentation.

    python experiments/K-T1/train.py --init kezek_tse.pt \
        --train-roots .../train-clean-100 .../train-clean-360 --val-root .../dev-clean \
        --rir-dirs .../RIRS_NOISES/simulated_rirs .../RIRS_NOISES/real_rirs_isotropic_noises \
        --noise-dirs .../musan/noise --music-dirs .../musan/music --out run

Every 5k steps: validation on 300 fixed simulated windows (dev-clean speakers). From 10k on, a
target-solo level off by more than 3 dB on two validations in a row stops the run (exit code 3):
that is a bug in the code or the loss, not a result. The evaluated checkpoint is the final one
(kezek_tse.pt); best.pt is kept for the report only.
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

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import WATCH_DB, WATCH_FROM, validate, watchdog_tripped  # noqa: E402
from kt1_data import ConversationDataset, build_index, collate, manifest  # noqa: E402

from kezek.augment import DomainBank  # noqa: E402
from kezek.losses import kt1_loss  # noqa: E402
from kezek.model import KezekTSE, SpeakerEncoder  # noqa: E402

WATCHDOG_EXIT = 3


def init_tracker(a, man: dict):
    """Airon run tracking (pip install airon; AIRON_URL, AIRON_API_KEY). A no-op if unavailable."""
    try:
        import airon

        return airon.init(
            project="kezek",
            name=a.run_name,
            experiment="K-T1",
            config={**vars(a), "manifest": {k: v for k, v in man.items() if k != "args"}},
            tags=["K-T1", "audio", "conversation"],
            total_steps=a.steps,
        )
    except Exception as e:
        print(f"airon tracking is off: {type(e).__name__}: {e}", flush=True)
        return None


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="K-T0c final checkpoint (state dict)")
    ap.add_argument("--train-roots", nargs="+", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--rir-dirs", nargs="*", default=[])
    ap.add_argument("--noise-dirs", nargs="*", default=[])
    ap.add_argument("--music-dirs", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="k-t1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--allow-overlap",
        action="store_true",
        help="dry runs only: skip the speaker-disjoint check",
    )
    ap.add_argument(
        "--resume", action="store_true", help="continue from <out>/last.pt if it exists"
    )
    ap.add_argument("--fast", action="store_true", help="cudnn.benchmark and pin_memory")
    ap.add_argument("--no-tracker", action="store_true")
    ap.add_argument(
        "--keep-every",
        type=int,
        default=0,
        help="also keep step_<N>.pt at these steps (diagnostics)",
    )
    ap.add_argument(
        "--stop-at", type=int, default=0, help="stop here; the LR schedule still spans --steps"
    )
    a = ap.parse_args()

    dev = a.device
    torch.backends.cudnn.benchmark = a.fast
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    train_idx = build_index(a.train_roots, f"{a.out}/train_index.json")
    val_idx = build_index([a.val_root], f"{a.out}/val_index.json")
    assert a.allow_overlap or not set(train_idx) & set(val_idx), "train/validation speakers overlap"
    bank = DomainBank.from_dirs(a.rir_dirs, a.noise_dirs, a.music_dirs)
    print(
        f"train: {len(train_idx)} speakers | val: {len(val_idx)} speakers | rirs {len(bank.rirs)} "
        f"noises {len(bank.noises)} music {len(bank.music)}",
        flush=True,
    )
    man = manifest(vars(a), bank, train_idx, val_idx, a.init)
    man_path = f"{a.out}/manifest.json"
    if os.path.exists(man_path) and a.resume:
        old = json.load(open(man_path))
        changed = [k for k in man if k != "args" and old.get(k) != man[k]]
        if changed:
            raise SystemExit(f"resume with different code or data: {changed} (see {man_path})")
    json.dump(man, open(man_path, "w"), indent=1)
    print(f"manifest: commit {man['commit']} dirty {man['dirty']}", flush=True)

    ckpt_path = f"{a.out}/last.pt"
    ckpt = (
        torch.load(ckpt_path, map_location="cpu")
        if a.resume and os.path.exists(ckpt_path)
        else None
    )
    start = ckpt["step"] if ckpt else 0

    train_set = ConversationDataset(train_idx, bank, a.steps * a.bs, seed=a.seed)
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_set, range(start * a.bs, len(train_set))),
        batch_size=a.bs,
        num_workers=a.workers,
        shuffle=False,
        drop_last=True,
        collate_fn=collate,
        prefetch_factor=4 if a.workers else None,
        pin_memory=a.fast,
    )
    val_set = ConversationDataset(val_idx, bank, a.n_val, seed=20_000)

    se = SpeakerEncoder(device=dev)
    model = KezekTSE().to(dev)
    model.load_state_dict(torch.load(a.init, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, a.warmup, a.steps))
    log, levels = [], []
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        log = ckpt["log"]
        levels = [(e["step"], e["all/target_solo_level_db"]) for e in log]
        print(f"resumed from step {start}", flush=True)

    batches = iter(train_dl)  # fork the workers before the tracker starts
    run = None if a.no_tracker else init_tracker(a, man)
    t0 = time.time()
    for step, b in enumerate(batches, start + 1):
        mix, clean = b["mix"].to(dev), b["clean"].to(dev)
        with torch.no_grad():
            d = se(b["enr_t"].to(dev))
        est = model(mix, d)
        loss, parts = kt1_loss(
            est, clean, b["act_t"].to(dev), b["silent"].to(dev), b["l_ref"].to(dev), d, se
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if run is not None and step % 10 == 0:
            run.log(
                {"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()}},
                step=step,
            )
        if step % 100 == 0 or step <= 10:
            print(
                f"step {step:6d}  loss {loss.item():7.3f}  {parts}  {time.time() - t0:6.0f}s",
                flush=True,
            )
        if step % a.eval_every == 0 or step in (a.steps, a.stop_at):
            m = validate(model, se, val_set, dev)
            print(
                f"EVAL step {step}: {json.dumps({k: round(v, 2) for k, v in m.items()})}",
                flush=True,
            )
            log.append({"step": step, **m})
            levels.append((step, m["all/target_solo_level_db"]))
            if run is not None:
                run.log({f"val/{k}": v for k, v in m.items()}, step=step)
            torch.save(model.state_dict(), f"{a.out}/kezek_tse.pt")
            if a.keep_every and step % a.keep_every == 0:
                torch.save(model.state_dict(), f"{a.out}/step_{step}.pt")
            score = m["all/overlap_sisdr"]
            if np.isfinite(score) and score >= max(e["all/overlap_sisdr"] for e in log):
                torch.save(model.state_dict(), f"{a.out}/best.pt")
            json.dump(log, open(f"{a.out}/eval_log.json", "w"), indent=1)
            state = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "log": log,
            }
            torch.save(state, f"{ckpt_path}.tmp")
            os.replace(f"{ckpt_path}.tmp", ckpt_path)
            if watchdog_tripped(levels):
                print(
                    f"WATCHDOG STOP: target solo level {levels[-2:]} outside В±{WATCH_DB} dB "
                    f"(from step {WATCH_FROM}); a code or loss bug, not a result",
                    flush=True,
                )
                if run is not None:
                    run.finish()
                sys.exit(WATCHDOG_EXIT)
        if step >= min(a.steps, a.stop_at or a.steps):
            break
    if run is not None:
        run.finish()
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
