"""Run one whole train mixture and one whole test mixture through a checkpoint, and save the audio to listen to.

    python experiments/K-T0/listen_two.py <run folder> <checkpoint> <out folder>
"""

import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from kezek import mixing
from kezek.model import KezekTSE, SpeakerEncoder, si_sdr_loss

SR = 16000
run_dir, ckpt, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)

se = SpeakerEncoder(device="cuda")
model = KezekTSE().to("cuda")
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()


def peak(x, p=0.9):
    return np.float32(x * p / max(np.abs(x).max(), 1e-6))


def run(tag, index_file, seed):
    # Same choice of speakers and utterances as MixDataset item 0, but whole files and 0 dB.
    index = json.load(open(f"{run_dir}/{index_file}"))
    names = sorted(index)
    rng = np.random.default_rng([seed, 0])
    a, b = rng.choice(len(names), 2, replace=False)
    ua, ub = index[names[a]], index[names[b]]
    ia, ib = rng.choice(len(ua), 2, replace=False), rng.choice(len(ub), 2, replace=False)
    tgt = sf.read(ua[ia[0]][0], dtype="float32")[0]
    itf = sf.read(ub[ib[0]][0], dtype="float32")[0]
    n = min(len(tgt), len(itf))
    mix, t, _ = mixing.mix_pair(tgt[:n], itf[:n], sir_db=0.0, offset=0)
    g = 0.9 / max(np.abs(mix).max(), 1e-6)
    mix, t = np.float32(mix * g), np.float32(t * g)
    enr = sf.read(ua[ia[1]][0], start=0, frames=6 * SR, dtype="float32")[0]

    with torch.no_grad():
        m = torch.from_numpy(mix)[None].cuda()
        r = torch.from_numpy(t)[None].cuda()
        est = model(m, se(torch.from_numpy(enr)[None].cuda()))
        mix_db = float(-si_sdr_loss(m, r).mean())
        est_db = float(-si_sdr_loss(est, r).mean())

    sf.write(f"{out}/{tag}_output.wav", peak(est[0].cpu().numpy()), SR)
    sf.write(f"{out}/{tag}_mixture.wav", peak(mix), SR)
    sf.write(f"{out}/{tag}_clean_target.wav", peak(t), SR)
    print(f"{tag}: target speaker {names[a]}, other speaker {names[b]}, {n / SR:.1f} s, "
          f"mixture {mix_db:.2f} dB, output {est_db:.2f} dB, improvement {est_db - mix_db:+.2f} dB")


run("train", "train_index.json", 0)
run("test", "test_index.json", 10_000)
