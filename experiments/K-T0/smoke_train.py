"""Smoke test of our own architecture on LibriSpeech dev-clean (toy scale, 40 speakers).

Not the K-T0 protocol: it only checks that the pipeline learns and that the model really
follows the voice sample. Speakers are split 32 train / 8 test, never mixed.

    python experiments/K-T0/smoke_train.py --root /workspace/data/librispeech/LibriSpeech/dev-clean
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

from kezek import mixing
from kezek.model import KezekTSE, SpeakerEncoder, kezek_loss, si_sdr_loss

SR = 16000


def load(root: str) -> dict[str, list[np.ndarray]]:
    spk: dict[str, list[np.ndarray]] = {}
    for f in sorted(glob.glob(f"{root}/*/*/*.flac")):
        x, sr = sf.read(f, dtype="float32")
        assert sr == SR
        if len(x) >= 7 * SR:  # long enough for a 4 s mixture crop and a 6 s enrollment crop
            spk.setdefault(os.path.basename(os.path.dirname(os.path.dirname(f))), []).append(x)
    return spk


def crop(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    s = int(rng.integers(0, len(x) - n + 1)) if len(x) > n else 0
    return x[s : s + n]


def sample(spk: dict[str, list[np.ndarray]], names: list[str], rng, seg=4.0, enr=6.0):
    """One training example: (mixture, target, interferer, enrollment of target, enrollment of interferer)."""
    a, b = rng.choice(len(names), 2, replace=False)
    ua, ub = spk[names[a]], spk[names[b]]
    ia, ib = rng.choice(len(ua), 2, replace=False), rng.choice(len(ub), 2, replace=False)
    n = int(seg * SR)
    tgt, itf = crop(ua[ia[0]], n, rng), crop(ub[ib[0]], n, rng)
    mix, t, i = mixing.mix_pair(tgt, itf, sir_db=float(rng.uniform(-5, 5)), offset=int(rng.integers(0, SR)))
    mix, t, i = mix[:n], t[:n], i[:n]
    g = 0.9 / max(np.abs(mix).max(), 1e-6)
    enr_t, enr_i = crop(ua[ia[1]], int(enr * SR), rng), crop(ub[ib[1]], int(enr * SR), rng)
    return [np.float32(v) for v in (mix * g, t * g, i * g, enr_t, enr_i)]


def batch(spk, names, rng, bs, dev):
    items = [sample(spk, names, rng) for _ in range(bs)]
    return [torch.from_numpy(np.stack(c)).to(dev) for c in zip(*items)]


@torch.no_grad()
def evaluate(model, se, spk, names, dev, n=200, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    res = {"mix": [], "est": [], "wrong_vs_target": [], "wrong_vs_interf": []}
    for _ in range(n // 20):
        mix, t, i, et, ei = batch(spk, names, rng, 20, dev)
        e_right = model(mix, se(et))
        e_wrong = model(mix, se(ei))  # sample of the OTHER speaker: should extract the interferer
        res["mix"] += (-si_sdr_loss(mix, t)).tolist()
        res["est"] += (-si_sdr_loss(e_right, t)).tolist()
        res["wrong_vs_target"] += (-si_sdr_loss(e_wrong, t)).tolist()
        res["wrong_vs_interf"] += (-si_sdr_loss(e_wrong, i)).tolist()
    m = {k: float(np.mean(v)) for k, v in res.items()}
    m["sisdri"] = m["est"] - m["mix"]
    # share of mixtures where the right sample gives the right speaker
    m["pick_acc"] = float(np.mean([
        (a > b) for a, b in zip(res["est"], res["wrong_vs_target"])]))
    model.train()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--spk-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="/workspace/ckpt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed)
    spk = load(a.root)
    names = sorted(spk)
    train_n, test_n = names[:-8], names[-8:]
    print(f"speakers: {len(train_n)} train / {len(test_n)} test (disjoint); "
          f"utts {sum(len(spk[n]) for n in names)}", flush=True)

    se = SpeakerEncoder(device=dev)
    model = KezekTSE().to(dev)
    print("params (M): %.2f" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    log = []
    for step in range(1, a.steps + 1):
        mix, t, _, et, _ = batch(spk, train_n, rng, a.bs, dev)
        with torch.no_grad():
            d = se(et)
        est = model(mix, d)
        loss, parts = kezek_loss(est, t, d, se, a.spk_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {float(loss):7.3f}  {parts}  {time.time()-t0:5.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, se, spk, test_n, dev)
            print(f"EVAL step {step}: {json.dumps({k: round(v, 2) for k, v in m.items()})}", flush=True)
            log.append({"step": step, **m})
            torch.save(model.state_dict(), f"{a.out}/kezek_tse.pt")
    json.dump(log, open(f"{a.out}/eval_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
