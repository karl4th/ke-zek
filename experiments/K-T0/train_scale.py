"""Train the audio Kezek TSE on many speakers (LibriSpeech train-clean-100/360), test on dev-clean.

Same mixture recipe and model as smoke_train.py; the only change is the number of training
speakers (32 -> ~900), to see whether generalization to unseen speakers appears. dev-clean
speakers are disjoint from the training sets by LibriSpeech construction.

    python experiments/K-T0/train_scale.py \
        --train-roots <.../train-clean-100> <.../train-clean-360> --test-root <.../dev-clean>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

from kezek import mixing
from kezek.model import KezekTSE, SpeakerEncoder, kezek_loss, si_sdr_loss

SR = 16000
MIN_S = 7.0  # long enough for a 4 s mixture crop and a 6 s enrollment crop


def build_index(roots: list[str], cache: str | None = None) -> dict[str, list[tuple[str, int]]]:
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    files = sorted(f for r in roots for f in glob.glob(f"{r}/*/*/*.flac"))

    def info(f):
        return f, sf.info(f).frames

    index: dict[str, list[tuple[str, int]]] = {}
    with ThreadPoolExecutor(32) as ex:
        for f, n in ex.map(info, files):
            if n >= MIN_S * SR:
                index.setdefault(os.path.basename(os.path.dirname(os.path.dirname(f))), []).append((f, n))
    index = {k: v for k, v in index.items() if len(v) >= 2}
    if cache:
        json.dump(index, open(cache, "w"))
    return index


class MixDataset(torch.utils.data.Dataset):
    """Mixture i is a deterministic function of (seed, i): reproducible train and test sets."""

    def __init__(self, index, n: int, seed: int, seg: float = 4.0, enr: float = 6.0):
        self.index, self.names = index, sorted(index)
        self.n, self.seed, self.seg, self.enr = n, seed, seg, enr

    def __len__(self):
        return self.n

    @staticmethod
    def _read(item, n, rng):
        path, total = item
        s = int(rng.integers(0, total - n + 1))
        return sf.read(path, start=s, frames=n, dtype="float32")[0]

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        a, b = rng.choice(len(self.names), 2, replace=False)
        ua, ub = self.index[self.names[a]], self.index[self.names[b]]
        ia, ib = rng.choice(len(ua), 2, replace=False), rng.choice(len(ub), 2, replace=False)
        n, ne = int(self.seg * SR), int(self.enr * SR)
        tgt, itf = self._read(ua[ia[0]], n, rng), self._read(ub[ib[0]], n, rng)
        mix, t, ifr = mixing.mix_pair(
            tgt, itf, sir_db=float(rng.uniform(-5, 5)), offset=int(rng.integers(0, SR))
        )
        mix, t, ifr = mix[:n], t[:n], ifr[:n]
        g = 0.9 / max(np.abs(mix).max(), 1e-6)
        enr_t, enr_i = self._read(ua[ia[1]], ne, rng), self._read(ub[ib[1]], ne, rng)
        return tuple(np.float32(v) for v in (mix * g, t * g, ifr * g, enr_t, enr_i))


@torch.no_grad()
def evaluate(model, se, loader, dev):
    model.eval()
    res = {"mix": [], "est": [], "wrong_vs_target": [], "wrong_vs_interf": []}
    for mix, t, i, et, ei in loader:
        mix, t, i, et, ei = (x.to(dev) for x in (mix, t, i, et, ei))
        e_right, e_wrong = model(mix, se(et)), model(mix, se(ei))
        res["mix"] += (-si_sdr_loss(mix, t)).tolist()
        res["est"] += (-si_sdr_loss(e_right, t)).tolist()
        res["wrong_vs_target"] += (-si_sdr_loss(e_wrong, t)).tolist()
        res["wrong_vs_interf"] += (-si_sdr_loss(e_wrong, i)).tolist()
    m = {k: float(np.mean(v)) for k, v in res.items()}
    m["sisdri"] = m["est"] - m["mix"]
    m["pick_acc"] = float(np.mean([a > b for a, b in zip(res["est"], res["wrong_vs_target"])]))
    model.train()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-roots", nargs="+", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--spk-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--out", default="/workspace/ckpt_scale")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    train_idx = build_index(a.train_roots, f"{a.out}/train_index.json")
    test_idx = build_index([a.test_root], f"{a.out}/test_index.json")
    assert not set(train_idx) & set(test_idx), "train/test speakers overlap"
    print(f"train: {len(train_idx)} speakers, {sum(len(v) for v in train_idx.values())} utts | "
          f"test: {len(test_idx)} speakers, {sum(len(v) for v in test_idx.values())} utts "
          f"(index {time.time()-t0:.0f}s)", flush=True)

    train_dl = torch.utils.data.DataLoader(
        MixDataset(train_idx, a.steps * a.bs, seed=a.seed), batch_size=a.bs, num_workers=a.workers,
        shuffle=False, drop_last=True, prefetch_factor=4,
    )
    test_dl = torch.utils.data.DataLoader(
        MixDataset(test_idx, a.n_test, seed=10_000), batch_size=20, num_workers=2, shuffle=False
    )

    se = SpeakerEncoder(device=dev)
    model = KezekTSE().to(dev)
    print("params (M): %.2f" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)

    log, t0 = [], time.time()
    for step, (mix, t, _, et, _) in enumerate(train_dl, 1):
        mix, t, et = mix.to(dev), t.to(dev), et.to(dev)
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
            print(f"step {step:6d}  loss {float(loss):7.3f}  {parts}  {time.time()-t0:6.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, se, test_dl, dev)
            print(f"EVAL step {step}: {json.dumps({k: round(v, 2) for k, v in m.items()})}", flush=True)
            log.append({"step": step, **m})
            torch.save(model.state_dict(), f"{a.out}/kezek_tse.pt")
            json.dump(log, open(f"{a.out}/eval_log.json", "w"), indent=1)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
