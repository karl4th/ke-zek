"""Render the validation recordings once (the same chain and seed per conversation as
``s1_eval_full.py --aug-root``) and keep them as 16-bit files, so evaluations read them instead of
rendering again: <out>/<conversation>_<part>.npy (int16, peak 0.9 as the evaluation feeds it).

    python experiments/K-M1/render_cache.py --data /data/oy16 --split-file split.json \
        --aug-root /data/aug --out /data/xcache [--shard 0 --shards 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
from rdata import _Conversation, splits  # noqa: E402
from s1_eval_full import recording  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--aug-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conversations", type=int, default=113)
    ap.add_argument("--max-minutes", type=float, default=100000.0)
    ap.add_argument("--render-part", choices=("all", "sources", "finish"), default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    from render import Render

    from kezek.augment import DomainBank

    r = a.aug_root
    render = Render(DomainBank.from_dirs(
        [f"{r}/RIRS_NOISES/simulated_rirs", f"{r}/RIRS_NOISES/real_rirs_isotropic_noises"],
        [f"{r}/musan/noise"], [f"{r}/musan/music"]))
    _, val, _ = splits(a.data, split_file=a.split_file or None)
    os.makedirs(a.out, exist_ok=True)
    for k, pair in enumerate(val[: a.conversations]):
        if k % a.shards != a.shard:
            continue
        f = f"{a.out}/{pair.conversation_id}_{a.render_part}.npy"
        if os.path.exists(f):
            continue
        t0 = time.time()
        c = _Conversation(pair)
        nfr = min(c.frames, int(a.max_minutes * 6000))
        sa, sb = c.audio(0, 0, nfr), c.audio(1, 0, nfr)
        act10 = np.stack([c.act[0][:nfr], c.act[1][:nfr]])
        x = recording(sa, sb, act10, k, render, a.render_part)
        np.save(f + ".tmp.npy", np.round(np.clip(x, -1, 1) * 32767).astype(np.int16))
        os.replace(f + ".tmp.npy", f)
        print(f"{k} {pair.conversation_id} {len(x) / 16000 / 60:.1f} min {time.time() - t0:.0f} s", flush=True)
    print("RENDER DONE", flush=True)


if __name__ == "__main__":
    main()
