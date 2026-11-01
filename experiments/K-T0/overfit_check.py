"""Overfit check: can the model memorize one fixed batch? (sanity test of model + loss)

If SI-SDR on this batch does not climb far above 20 dB, there is a bug in the model or the loss,
not a data problem. Run after smoke_train.py (it reuses its data helpers):

    python experiments/K-T0/overfit_check.py --root /workspace/data/librispeech/LibriSpeech/dev-clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from smoke_train import batch, load  # noqa: E402

from kezek.model import KezekTSE, SpeakerEncoder, kezek_loss, si_sdr_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--spk-weight", type=float, default=0.1)
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    spk = load(a.root)
    names = sorted(spk)[:-8]  # training speakers only
    mix, t, _, et, _ = batch(spk, names, np.random.default_rng(0), a.bs, dev)  # ONE fixed batch
    se = SpeakerEncoder(device=dev)
    with torch.no_grad():
        d = se(et)
    model = KezekTSE().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    print("mixture SI-SDR on this batch: %.2f dB" % float((-si_sdr_loss(mix, t)).mean()))
    for step in range(1, a.steps + 1):
        est = model(mix, d)
        loss, parts = kezek_loss(est, t, d, se, a.spk_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 25 == 0:
            print(f"step {step:4d}  SI-SDR {-parts['neg_sisdr']:6.2f} dB  spk {parts['spk']:.3f}", flush=True)


if __name__ == "__main__":
    main()
