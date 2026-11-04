"""Segment-wise validation of any K-R1 extractor (the K-T1 metrics, ``model(mix, enr, d)``)."""

from __future__ import annotations

import numpy as np
import torch

from kezek import metrics


@torch.no_grad()
def validate(model, se, dataset, device: str, batch: int = 16, ci: bool = False) -> dict:
    """Gate metrics pooled over windows, split by the sign of the window SIR; speaker check
    (wrong_speaker, pick_acc) on windows where the target is active >= 1.5 s."""
    was_training = model.training
    model.eval()
    groups: dict[str, list[dict]] = {"all": [], "sir_neg": [], "sir_pos": []}
    cos_t, cos_o, act_n, pick = [], [], [], []
    for i0 in range(0, len(dataset), batch):
        items = [dataset[i] for i in range(i0, min(i0 + batch, len(dataset)))]
        mix = torch.from_numpy(np.stack([it["mix"] for it in items])).to(device)
        n_t = min(len(it["enr_t"]) for it in items)
        n_o = min(len(it["enr_o"]) for it in items)
        et = torch.from_numpy(np.stack([it["enr_t"][:n_t] for it in items])).to(device)
        eo = torch.from_numpy(np.stack([it["enr_o"][:n_o] for it in items])).to(device)
        dt, do = se(et), se(eo)
        est, est_wrong = model(mix, et, dt), model(mix, eo, do)
        e_out = se(est)
        ct = torch.nn.functional.cosine_similarity(e_out, dt, dim=-1).cpu().numpy()
        co = torch.nn.functional.cosine_similarity(e_out, do, dim=-1).cpu().numpy()
        est, est_wrong = est.cpu().numpy(), est_wrong.cpu().numpy()
        for k, it in enumerate(items):
            s = metrics.segment_scores(
                est[k], it["clean"], it["other"], it["mix"], it["act_t"], it["act_o"],
                it["disputed"], float(it["l_ref"]),
            )
            groups["all"].append(s)
            groups["sir_neg" if it["sir_db"] < 0 else "sir_pos"].append(s)
            cos_t.append(ct[k])
            cos_o.append(co[k])
            act_n.append(int(it["act_t"].sum()))
            pick.append(metrics.si_sdr(est[k], it["clean"]) > metrics.si_sdr(est_wrong[k], it["clean"]))
    model.train(was_training)
    out = {f"{g}/{k}": v for g, sc in groups.items() if sc for k, v in metrics.aggregate(sc).items()}
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
    if ci:
        for g, sc in groups.items():
            if sc:
                out.update({f"{g}/{k}_ci": v for k, v in metrics.bootstrap_ci(sc).items()})
        out["wrong_speaker_ci"] = metrics.share_ci((np.array(cos_t) < np.array(cos_o))[ok])
        out["pick_acc_ci"] = metrics.share_ci(np.array(pick)[ok])
    return out
