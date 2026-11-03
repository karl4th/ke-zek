"""K-T1 overfit check: one batch of simulated windows, many steps from the K-T0c checkpoint.

The loss must go to its ceiling (SNR on active frames -> ~30 dB, silence penalty -> 0) and the
target-solo level on that batch must reach 0 +- 0.5 dB.

    python experiments/K-T1/overfit_check.py --init kezek_tse.pt --root .../dev-clean
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kt1_data import ConversationDataset, build_index, collate  # noqa: E402

from kezek import metrics  # noqa: E402
from kezek.augment import DomainBank  # noqa: E402
from kezek.losses import kt1_loss  # noqa: E402
from kezek.model import KezekTSE, SpeakerEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--rir-dirs", nargs="*", default=[])
    ap.add_argument("--noise-dirs", nargs="*", default=[])
    ap.add_argument("--music-dirs", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    bank = DomainBank.from_dirs(a.rir_dirs, a.noise_dirs, a.music_dirs)
    ds = ConversationDataset(build_index([a.root]), bank, a.bs, seed=123)
    items = [ds[i] for i in range(a.bs)]
    b = {k: v.to(a.device) for k, v in collate(items).items()}
    se = SpeakerEncoder(device=a.device)
    model = KezekTSE().to(a.device)
    model.load_state_dict(torch.load(a.init, map_location="cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    with torch.no_grad():
        d = se(b["enr_t"])

    def level():
        with torch.no_grad():
            est = model(b["mix"], d).cpu().numpy()
        sc = [
            metrics.segment_scores(
                est[k],
                it["clean"],
                it["other"],
                it["mix"],
                it["act_t"],
                it["act_o"],
                it["disputed"],
                float(it["l_ref"]),
            )
            for k, it in enumerate(items)
        ]
        return metrics.aggregate(sc)

    m0 = level()
    lvl, leak = m0["target_solo_level_db"], m0["other_solo_leak_db"]
    print(f"start: level {lvl:+.2f} dB, leak {leak:+.1f} dB", flush=True)
    for step in range(1, a.steps + 1):
        loss, parts = kt1_loss(
            model(b["mix"], d), b["clean"], b["act_t"], b["silent"], b["l_ref"], d, se
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d} loss {loss.item():8.3f} {parts}", flush=True)
    m = level()
    print({k: round(v, 2) for k, v in m.items()})
    ok = (
        m["target_solo_level_db"] == m["target_solo_level_db"]
        and abs(m["target_solo_level_db"]) <= 0.5
    )
    print("OVERFIT", "OK" if ok and parts["snr_active"] > 25 else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
