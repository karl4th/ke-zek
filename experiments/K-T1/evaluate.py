"""K-T1 evaluation by segment type.

Without ``--final`` this script only runs the validation set (simulated windows on dev-clean
speakers) and refuses to open any test list. Test lists are read only with ``--final``.

    python experiments/K-T1/evaluate.py --checkpoint kezek_tse.pt --val-root .../dev-clean \
        [--rir-dirs ...] [--noise-dirs ...] [--music-dirs ...]
    python experiments/K-T1/evaluate.py --checkpoint kezek_tse.pt --final --tests tests.json

A test list is JSON: {"name": [item_dir, ...]}. An item dir is what ``kezek prepare`` writes
(mixture.wav, target.wav, interferer.wav, enrollment.wav); the clean tracks are the
per-speaker channels of a real conversation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from kezek import metrics
from kezek.augment import DomainBank
from kezek.enrollment import pick_enrollment
from kezek.infer import extract_long
from kezek.mixing import FRAME_S_CONV, active_frames, frame_count, reference_level

SR = 16000
FRAME = int(FRAME_S_CONV * SR)
WATCH_FROM, WATCH_DB = 10_000, 3.0


def watchdog_tripped(
    history: list[tuple[int, float]], from_step: int = WATCH_FROM, limit_db: float = WATCH_DB
) -> bool:
    """Stop when |target solo level| > limit on two validations in a row, from ``from_step``.
    ``history``: (step, target_solo_level_db) of every validation so far."""
    h = [(s, v) for s, v in history if s >= from_step]
    return len(h) >= 2 and all(abs(v) > limit_db for _, v in h[-2:])


@torch.no_grad()
def validate(model, se, dataset, device: str, batch: int = 20) -> dict[str, float]:
    """300 fixed simulated windows. Gate metrics pooled; split by clean/augmented window and by
    the sign of SIR; speaker check on windows with the target active >= 1.5 s."""
    model.eval()
    groups: dict[str, list[dict]] = {
        "all": [],
        "clean": [],
        "aug": [],
        "sir_neg": [],
        "sir_pos": [],
    }
    cos_t, cos_o, act_n, pick = [], [], [], []
    for i0 in range(0, len(dataset), batch):
        items = [dataset[i] for i in range(i0, min(i0 + batch, len(dataset)))]
        n_enr = min(len(it["enr_t"]) for it in items)
        mix = torch.from_numpy(np.stack([it["mix"] for it in items])).to(device)
        et = torch.from_numpy(np.stack([it["enr_t"][:n_enr] for it in items])).to(device)
        n_eo = min(len(it["enr_o"]) for it in items)
        eo = torch.from_numpy(np.stack([it["enr_o"][:n_eo] for it in items])).to(device)
        dt, do = se(et), se(eo)
        est, est_wrong = model(mix, dt), model(mix, do)
        e_out = se(est)
        ct = torch.nn.functional.cosine_similarity(e_out, dt, dim=-1).cpu().numpy()
        co = torch.nn.functional.cosine_similarity(e_out, do, dim=-1).cpu().numpy()
        for k, it in enumerate(items):
            y = est[k].cpu().numpy()
            s = metrics.segment_scores(
                y,
                it["clean"],
                it["other"],
                it["mix"],
                it["act_t"],
                it["act_o"],
                it["disputed"],
                float(it["l_ref"]),
            )
            groups["all"].append(s)
            groups["clean" if it["clean_window"] else "aug"].append(s)
            groups["sir_neg" if it["sir_db"] < 0 else "sir_pos"].append(s)
            cos_t.append(ct[k])
            cos_o.append(co[k])
            act_n.append(int(it["act_t"].sum()))
            pick.append(
                metrics.si_sdr(y, it["clean"])
                > metrics.si_sdr(est_wrong[k].cpu().numpy(), it["clean"])
            )
    model.train()
    out = {
        f"{g}/{k}": v for g, sc in groups.items() if sc for k, v in metrics.aggregate(sc).items()
    }
    share, n_used, n_skip = metrics.wrong_speaker(np.array(cos_t), np.array(cos_o), np.array(act_n))
    ok = np.array(act_n) >= 150
    out.update(
        {
            "wrong_speaker": share,
            "speaker_windows": n_used,
            "speaker_windows_skipped": n_skip,
            "pick_acc": float(np.mean(np.array(pick)[ok])) if ok.any() else float("nan"),
        }
    )
    return out


def labels_from_tracks(target: np.ndarray, other: np.ndarray, disputed_s: float = 0.15):
    """Frame labels for a real conversation from its clean per-speaker tracks: energy VAD, and
    disputed frames around every run boundary (gaps < 0.3 s closed first)."""
    a_t, a_o = active_frames(target, SR), active_frames(other, SR)
    w = int(round(disputed_s / FRAME_S_CONV))
    disputed = np.zeros(len(a_t), bool)
    for a in (a_t, a_o):
        b = a.copy()
        idx = np.flatnonzero(b)
        for i, j in zip(idx[:-1], idx[1:], strict=False):
            if 1 < j - i <= 30:
                b[i:j] = True
        for e in np.flatnonzero(np.diff(b.astype(int))) + 1:
            disputed[max(0, e - w) : e + w + 1] = True
    return a_t, a_o, disputed


@torch.no_grad()
def score_item(model, se, item_dir: str, device: str) -> tuple[dict, dict]:
    read = lambda n: sf.read(os.path.join(item_dir, f"{n}.wav"), dtype="float32")[0]  # noqa: E731
    mix, tgt, oth, enr = (read(n) for n in ("mixture", "target", "interferer", "enrollment"))
    n = frame_count(len(mix), SR) * FRAME
    mix, tgt, oth = mix[:n], tgt[:n], oth[:n]
    a_t, a_o, disp = labels_from_tracks(tgt, oth)
    d = se(torch.from_numpy(enr)[None].to(device))
    est = extract_long(model, mix, d)
    s = metrics.segment_scores(
        est, tgt, oth, mix, a_t, a_o, disp, reference_level(tgt, a_t, enr, SR)
    )
    # speaker check on 4 s windows of this recording, with an enrollment of the other speaker
    solo_o = a_o & ~a_t
    spk = {"cos_t": [], "cos_o": [], "act": []}
    try:
        eo, _ = pick_enrollment(mix, solo_o, SR, duration_s=6.0, frame_s=FRAME_S_CONV)
        do = se(torch.from_numpy(np.float32(eo))[None].to(device))
        win = 4 * SR
        for st in range(0, n - win + 1, 3 * SR):
            e = se(torch.from_numpy(np.float32(est[st : st + win]))[None].to(device))
            spk["cos_t"].append(float(torch.nn.functional.cosine_similarity(e, d)[0]))
            spk["cos_o"].append(float(torch.nn.functional.cosine_similarity(e, do)[0]))
            spk["act"].append(int(a_t[st // FRAME : (st + win) // FRAME].sum()))
    except ValueError:
        pass  # not enough solo of the other speaker for an enrollment
    return s, spk


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-root", help="dev-clean, for the validation set")
    ap.add_argument("--rir-dirs", nargs="*", default=[])
    ap.add_argument("--noise-dirs", nargs="*", default=[])
    ap.add_argument("--music-dirs", nargs="*", default=[])
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument(
        "--final", action="store_true", help="read the test lists (only for the final evaluation)"
    )
    ap.add_argument("--tests", help="JSON test list; read only with --final")
    ap.add_argument("--out", help="write the result as JSON here")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.tests and not a.final:
        ap.error("test lists are read only with --final")
    if not a.final and not a.val_root:
        ap.error("--val-root is required without --final")
    return a


def main(argv=None):
    a = parse(argv)
    from kezek.model import KezekTSE, SpeakerEncoder

    se = SpeakerEncoder(device=a.device)
    model = KezekTSE().to(a.device)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    model.eval()
    if not a.final:
        sys.path.insert(0, os.path.dirname(__file__))
        from kt1_data import ConversationDataset, build_index

        bank = DomainBank.from_dirs(a.rir_dirs, a.noise_dirs, a.music_dirs)
        val = ConversationDataset(build_index([a.val_root]), bank, a.n_val, seed=20_000)
        res = {"validation": validate(model, se, val, a.device)}
    else:
        res = {}
        for name, items in json.load(open(a.tests)).items():
            sc, spk = zip(*(score_item(model, se, d, a.device) for d in items), strict=True)
            r = metrics.aggregate(list(sc))
            ws = metrics.wrong_speaker(
                np.concatenate([s["cos_t"] for s in spk]) if spk else [],
                np.concatenate([s["cos_o"] for s in spk]) if spk else [],
                np.concatenate([s["act"] for s in spk]) if spk else [],
            )
            r.update(
                {"wrong_speaker": ws[0], "speaker_windows": ws[1], "speaker_windows_skipped": ws[2]}
            )
            res[name] = r
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
