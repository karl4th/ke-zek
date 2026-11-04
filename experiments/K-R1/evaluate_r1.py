"""Score a KezekTSE checkpoint, or the unprocessed mixture, on fixed Open Yap windows.

    python experiments/K-R1/evaluate_r1.py --data /data/oy16 --model mixture --out m.json
    python experiments/K-R1/evaluate_r1.py --data /data/oy16 --model run/kezek_tse.pt --out r.json
    python experiments/K-R1/evaluate_r1.py ... --split test --final

The windows are the same ``nat`` and ``ov`` sets as in training validation (same seeds), so
numbers from different models are paired. The test split is read only with ``--final``.
Bootstrap 95% intervals over windows are included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-T1"))
from evaluate import validate  # noqa: E402
from rdata import RealWindows, splits  # noqa: E402

from kezek.model import KezekTSE, SpeakerEncoder  # noqa: E402


class Mixture(torch.nn.Module):
    def forward(self, mix, d):
        return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="'mixture' or a KezekTSE state dict")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--final", action="store_true", help="required to read the test split")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        raise SystemExit("the test split is read only with --final")
    _, val, test = splits(a.data)
    pairs = val if a.split == "val" else test
    seeds = (20_000, 30_000) if a.split == "val" else (40_000, 50_000)
    sets = {
        "nat": RealWindows(pairs, a.n, seed=seeds[0], p_overlap=0.0, peak=(0.9, 0.9)),
        "ov": RealWindows(pairs, a.n, seed=seeds[1], p_overlap=1.0, peak=(0.9, 0.9)),
    }
    se = SpeakerEncoder(device=a.device)
    if a.model == "mixture":
        model = Mixture()
    else:
        model = KezekTSE().to(a.device)
        model.load_state_dict(torch.load(a.model, map_location="cpu"))
    out = {"model": a.model, "split": a.split, "n": a.n, "conversations": len(pairs)}
    with torch.no_grad():
        for name, ds in sets.items():
            out.update({f"{name}/{k}": v for k, v in validate(model, se, ds, a.device, ci=True).items()})
    json.dump(out, open(a.out, "w"), indent=1)
    keys = ("all/overlap_sisdri", "all/target_solo_snr", "all/target_solo_level_db",
            "all/other_solo_leak_db", "all/both_silent_db", "all/backchannel_recall", "pick_acc")
    print(json.dumps({f"{n}/{k}": round(out[f"{n}/{k}"], 3) for n in sets for k in keys}))


if __name__ == "__main__":
    main()
