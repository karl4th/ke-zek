"""Kezek v3 stage 1 on whole recordings: identity pass in windows, slot memory, activity of
memory slots A and B (m2s_infer), scored against the Open Yap labels.

    python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file split.json \
        --model run/model.pt --conversations 40 --out s1_val.json [--real-clip clip.wav]

The A/B mapping is chosen ONCE per conversation (the order with more agreeing active frames).
Reported, pooled over conversations:
* activity precision / recall / F1 of the memory slots (20 ms frames, threshold 0.5);
* short-reply recall: runs < 1 s next to the other person's speech, found when >= 50% of their
  frames are active in the right slot;
* swaps: 4 s windows where both speak >= 0.5 s and the swapped mapping agrees on more active
  frames (by a margin of 20% of the window's speech); per conversation, per hour and the share of
  conversations without any; windows the memory flags uncertain are counted separately;
* with ``--real-clip`` (one person talking, no labels): share of speech frames in the dominant
  slot and in the other one.
The test split is read only with --final.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
import tracker  # noqa: E402
from m2s_data import backchannel_frames  # noqa: E402
from m2s_infer import build_memory, identity_pass, link_windows, load, memory_activity, relink_segments  # noqa: E402
from m2s_model import frames_20ms  # noqa: E402
from rdata import _Conversation, _runs, splits  # noqa: E402

SR, FRAME = 16_000, 160


def score_conversation(p_act: np.ndarray, act: np.ndarray, bc: np.ndarray, uncertain: np.ndarray):
    """p_act [2, F] memory activity (bool), act/bc [2, F] labels on the same 20 ms grid."""
    agree = [(p_act[0] & act[0]).sum() + (p_act[1] & act[1]).sum(),
             (p_act[0] & act[1]).sum() + (p_act[1] & act[0]).sum()]
    order = (0, 1) if agree[0] >= agree[1] else (1, 0)  # memory slot of speaker a, b
    p = p_act[list(order)]
    r = {"tp": float((p & act).sum()), "fp": float((p & ~act).sum()), "fn": float((~p & act).sum())}
    # collar: frames within 200 ms of a turn boundary of that person are not scored
    edge = np.zeros_like(act)
    for s in (0, 1):
        ch = np.flatnonzero(np.diff(act[s].astype(np.int8))) + 1
        for c in ch:
            edge[s, max(0, c - 10):c + 10] = True
    sc = ~edge
    r.update({"tp_c": float((p & act & sc).sum()), "fp_c": float((p & ~act & sc).sum()), "fn_c": float((~p & act & sc).sum())})
    # where false positives fall: near a boundary of that person, while only the other talks,
    # in silence of both, away from boundaries
    fp = p & ~act
    other = act[::-1]
    r.update({"fp_edge": float((fp & edge).sum()), "fp_other": float((fp & ~edge & other).sum()),
              "fp_silence": float((fp & ~edge & ~other).sum())})
    hit = hit_any = n = 0
    for s in (0, 1):
        for r0, r1 in _runs(bc[s]):
            n += 1
            hit += int(p[s, r0:r1].mean() >= 0.5)
            hit_any += int((p[0, r0:r1] | p[1, r0:r1]).mean() >= 0.5)
    r.update({"bc_hit": hit, "bc_n": n, "bc_hit_any": hit_any})
    win = 200  # 4 s on the 20 ms grid
    sw = sw_flag = n_win = 0
    for f0 in range(0, act.shape[1] - win + 1, win):
        a = act[:, f0:f0 + win]
        if a[0].sum() < 25 or a[1].sum() < 25:
            continue
        q = p[:, f0:f0 + win]
        keep = (q[0] & a[0]).sum() + (q[1] & a[1]).sum()
        swap = (q[0] & a[1]).sum() + (q[1] & a[0]).sum()
        n_win += 1
        if swap > keep + 0.2 * a.sum():
            sw += 1
            sw_flag += int(uncertain[f0:f0 + win].any())
    r.update({"swap_windows": sw, "swap_windows_flagged": sw_flag, "two_speaker_windows": n_win})
    return r


def render_part(render, rec, sa, sb, mask, rng, part="all"):
    """One recording chain on clean tracks: the whole chain, only the per-voice part (room, EQ,
    bleed, telephone band, compression) or only the mixture part (noise, music, gain, clipping, codec)."""
    if part == "all" or rec.chain.clean:
        return render.apply(rec, sa, sb, mask, rng)[2]
    if part == "sources":
        ra, rb = rec.sources(sa)[: len(sa)], rec.sources(sb)[: len(sb)]
        return (ra + rb) * rec.compression_gain(ra + rb)
    m = sa + sb
    e = float(np.mean(m[mask] ** 2)) if mask.any() else float(np.mean(m**2)) + 1e-12
    x, _ = rec.chain.finish(m, e, rng)
    return np.pad(x[: len(sa)], (0, max(0, len(sa) - len(x))))


@torch.no_grad()
def tsvad_pass(model, heads, x, windows, cents, device, iters=1, prof_s=60.0):
    """Activity of A and B [2, F20] from the profile head on every window (A, B = the linked
    recording centroids), averaged where windows overlap. With ``iters`` > 1 the profiles are then
    taken again from the solo speech of A and of B over the whole recording (up to ``prof_s``
    seconds each, spread over it) and the head runs again. Each window's p and assign are set to
    the fixed A/B order so the window metrics see this activity."""
    from m2s_model import segment_embeddings, similarity

    spk = model.enc._speaker[0]
    nf = len(x) // 320 + 1
    z = torch.stack([F.normalize(c.float(), dim=-1) for c in cents])[None].to(device)
    feats = []
    for w in windows:
        seg = torch.from_numpy(np.ascontiguousarray(x[w["start"]:w["end"]]))[None].to(device)
        hs = {id(getattr(hd, "enc", model.enc)): None for hd in heads}
        for k in hs:
            enc = next(getattr(hd, "enc", model.enc) for hd in heads if id(getattr(hd, "enc", model.enc)) == k)
            hs[k] = enc.frames(seg, return_hidden=True).half()
        feats.append((hs, {sc: segment_embeddings(spk, seg, *sc) for sc in {sc for hd in heads for sc in hd.scales}}))
    for it in range(iters):
        acc, cnt, ps = np.zeros((2, nf), np.float32), np.zeros(nf, np.float32), []
        for w, (hs, e) in zip(windows, feats):
            nfw = next(iter(hs.values())).shape[1]
            sims = {sc: similarity(emb, z, nfw) for sc, emb in e.items()}
            p = torch.stack([torch.sigmoid(hd(hs[id(getattr(hd, "enc", model.enc))].float(), z,
                                              torch.stack([sims[sc] for sc in hd.scales], 1) if hd.scales else None))[0]
                             for hd in heads]).mean(0).float().cpu().numpy()
            f0 = w["start"] // 320
            L = min(p.shape[1], nf - f0)
            acc[:, f0:f0 + L] += p[:, :L]
            cnt[f0:f0 + L] += 1
            ps.append(p)
        act = acc / np.maximum(cnt, 1)
        if it + 1 < iters:
            new = []
            for s in (0, 1):
                fr = np.flatnonzero((act[s] > 0.5) & (act[1 - s] < 0.5))
                if len(fr) < 100:  # under 2 s of solo: keep the linked profile
                    new.append(z[0, s])
                    continue
                fr = fr[np.linspace(0, len(fr) - 1, min(len(fr), int(prof_s * 50))).astype(int)]
                idx = (fr[:, None] * 320 + np.arange(320)[None]).ravel()
                idx = idx[idx < len(x)]
                y = torch.from_numpy(np.ascontiguousarray(x[idx]))[None].to(device).float()
                new.append(F.normalize(spk(y).float(), dim=-1)[0])
            z = torch.stack(new)[None]
    for w, p in zip(windows, ps):
        w["z"] = w["z"][list(w["assign"])]  # slot embeddings in A/B order too (identity metric)
        w["p"], w["assign"] = torch.from_numpy(p), [0, 1]
    return act


@torch.no_grad()
def references(tracks, act, spk, device, cap_s=30.0):
    """Clean reference embeddings of each speaker from their own track, one per half of the
    recording (labelled solo frames, up to ``cap_s``): [half, speaker, D]; NaN without 1 s of solo."""
    nf = act.shape[1]
    out = np.full((2, 2, spk.dim), np.nan, np.float32)
    for h, (f0, f1) in enumerate(((0, nf // 2), (nf // 2, nf))):
        for s in (0, 1):
            fr = (np.flatnonzero(act[s, f0:f1] & ~act[1 - s, f0:f1]) + f0)[: int(cap_s * 50)]
            idx = (fr[:, None] * 320 + np.arange(320)[None]).ravel()
            idx = idx[idx < len(tracks[s])]
            if len(idx) >= 16_000:
                z = spk(torch.from_numpy(np.ascontiguousarray(tracks[s][idx]))[None].to(device).float())
                out[h, s] = F.normalize(z.float().cpu(), dim=-1)[0].numpy()
    return out


def window_metrics(windows, p_act, act, refs, min_solo=2.0):
    """Per 16 s window after linking. A window is clear when both people have >= ``min_solo`` s of
    labelled solo in it; its slot order then follows the labels (agreement of its slot activity),
    and it is swapped when the linked order puts its slots on the other recording-level profiles.
    Unclear windows with speech: whether their activity lands on the wrong profile. Identity: do
    the slot embeddings match their own speaker's clean reference (other half of the recording)
    better than the other's (K-M1 id_accuracy)."""
    nf = act.shape[1]
    keep = (p_act[0] & act[0]).sum() + (p_act[1] & act[1]).sum()
    cross = (p_act[0] & act[1]).sum() + (p_act[1] & act[0]).sum()
    spk_of_mem = (0, 1) if keep >= cross else (1, 0)  # speaker held by memory A, B
    n, rows = len(windows), []
    for t, w in enumerate(windows):
        f0 = w["start"] // 320
        q = w["p"].float().numpy() > 0.5
        L = min(q.shape[1], nf - f0)
        if L <= 0:
            continue
        q, l = q[:, :L], act[:, f0:f0 + L]
        ref_solo = [float((l[s] & ~l[1 - s]).sum() * 0.02) for s in (0, 1)]
        k = (q[0] & l[0]).sum() + (q[1] & l[1]).sum()
        x = (q[0] & l[1]).sum() + (q[1] & l[0]).sum()
        spk_slot0 = 0 if k >= x else 1  # speaker in slot 0 by the labels
        linked_slot0 = spk_of_mem[0] if w["assign"][0] == 0 else spk_of_mem[1]
        clear = min(ref_solo) >= min_solo
        r = {"t": t, "clear": clear, "edge": t < 2 or t >= n - 2, "ref_solo_min": round(min(ref_solo), 2),
             "pred_solo_min": round(float(min(w["solo_s"])), 2), "margin": round(float(w["margin"]), 3)}
        if clear:
            r["swap"] = spk_slot0 != linked_slot0
            hc = 0 if f0 + L / 2 < nf / 2 else 1
            ref = refs[1 - hc]
            if not np.isnan(ref).any():
                z = F.normalize(w["z"].float(), dim=-1).numpy()
                sp = (spk_slot0, 1 - spk_slot0)
                same = z[0] @ ref[sp[0]] + z[1] @ ref[sp[1]]
                diff = z[0] @ ref[sp[1]] + z[1] @ ref[sp[0]]
                r["id_ok"] = bool(same > diff)
        elif k + x >= 25:
            r["wrong_profile"] = spk_slot0 != linked_slot0
        rows.append(r)
    runs, cur = [], 0
    for r in rows:
        if r.get("swap"):
            cur += 1
        elif r.get("swap") is False and cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return rows, runs


def decode_windows(windows, mode, overlap_thr=0.8):
    """Binary slot activity per window, written back to w["p"] (so solo, embeddings and linking
    follow it): the argmax of the powerset softmax (an overlap only where 'both' wins), or the
    marginals with an overlap kept only where the weaker slot is above ``overlap_thr``."""
    for w in windows:
        if mode == "powerset":
            if "ps" not in w:
                raise SystemExit("powerset decoding needs windows from an identity pass that kept 'ps'")
            c = w["ps"].float().argmax(0)
            w["p"] = torch.stack([(c == 1) | (c == 3), (c == 2) | (c == 3)]).float()
        else:
            p = w["p"].float().clone()
            both = (p[0] > 0.5) & (p[1] > 0.5) & (p.min(0).values < overlap_thr)
            weak = (p[1] < p[0]).long()  # the weaker slot in each frame
            p[1][both & (weak == 1)] = 0.0
            p[0][both & (weak == 0)] = 0.0
            w["p"] = p


@torch.no_grad()
def reembed(windows, x, spk, device, lab, mode, thr):
    """Slot embeddings from packed solo audio of each window, by the predicted solo (p_a (1 - p_b)
    above ``thr``) or by the labels (the speaker a slot holds: the one its activity agrees with in
    that window). ``label_full`` also sets the slot activity to the labels."""
    nf = lab.shape[1]
    for w in windows:
        f0 = w["start"] // 320
        p = w["p"].float().numpy()
        L = min(p.shape[1], nf - f0)
        if mode == "pred":
            solo = np.stack([p[0] * (1 - p[1]) > thr, p[1] * (1 - p[0]) > thr])[:, :L]
        else:
            l = lab[:, f0:f0 + L]
            q = p[:, :L] > 0.5
            keep = (q[0] & l[0]).sum() + (q[1] & l[1]).sum()
            cross = (q[0] & l[1]).sum() + (q[1] & l[0]).sum()
            sp = (0, 1) if keep >= cross else (1, 0)  # speaker held by slot 0, 1
            solo = np.stack([l[sp[0]] & ~l[sp[1]], l[sp[1]] & ~l[sp[0]]])
            if mode == "label_full":
                full = np.zeros_like(p)
                full[:, :L] = np.stack([l[sp[0]], l[sp[1]]])
                w["p"] = torch.from_numpy(full)
        rows, secs = [], []
        for s in (0, 1):
            fr = np.flatnonzero(solo[s]) + f0
            idx = (fr[:, None] * 320 + np.arange(320)[None]).ravel()
            idx = idx[idx < len(x)]
            secs.append(len(fr) * 0.02)
            if len(idx) < 1600:  # < 0.1 s: the masked window as in m2s_model._packed
                seg = np.zeros(int(w["end"] - w["start"]), np.float32)
                m = np.repeat(solo[s], 320)[: len(seg)]
                seg[: len(m)] = x[w["start"]:w["start"] + len(m)] * m
                rows.append(seg)
            else:
                rows.append(x[idx])
        Lm = max(len(r) for r in rows)
        batch = torch.stack([F.pad(torch.from_numpy(np.ascontiguousarray(r)), (0, Lm - len(r))) for r in rows]).to(device)
        rel = torch.tensor([len(r) / Lm for r in rows], device=device)
        w["z"] = F.normalize(spk(batch.float(), rel).float().cpu(), dim=-1)
        w["solo_s"] = secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=("val", "test", "train"), default="val")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--conversations", type=int, default=40)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--win-s", type=float, default=16.0)
    ap.add_argument("--hop-s", type=float, default=8.0)
    ap.add_argument("--aug-root", default="", help="render each conversation as one mono recording")
    ap.add_argument("--real-clip", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--linker", choices=("memory", "global"), default="global",
                    help="greedy slot memory or global Viterbi linking (m2s_infer.link_windows)")
    ap.add_argument("--cache", default="", help="directory of identity-pass windows per conversation "
                    "(written on the first run, reused after: relinking needs no model)")
    ap.add_argument("--combine", choices=("max", "mean", "center"), default="max",
                    help="overlapping windows: maximum or average of slot activity")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--link-params", default="{}", help="JSON keyword arguments of link_windows")
    ap.add_argument("--relink", default="", help="JSON keyword arguments of relink_segments (second "
                    "linking pass on long solo audio per segment); empty: off")
    ap.add_argument("--tag", default="", help="suffix of the tracker run name")
    ap.add_argument("--reembed", choices=("", "pred", "label", "label_full"), default="",
                    help="slot embeddings again from packed solo audio before linking: 'pred' from the "
                    "predicted solo above --solo-thr, 'label' from the labelled solo of the speaker each "
                    "slot holds (oracle embeddings), 'label_full' also replaces the slot activity by the labels")
    ap.add_argument("--solo-thr", type=float, default=0.5)
    ap.add_argument("--decode", choices=("marginal", "powerset", "overlap"), default="marginal",
                    help="slot activity of each window before linking: the marginals as they are, the most "
                    "likely powerset class (silence, A, B, both; needs windows with 'ps'), or the marginals "
                    "with overlap kept only where the weaker slot is above --overlap-thr; with powerset or "
                    "overlap the slot embeddings are taken again from the new solo")
    ap.add_argument("--overlap-thr", type=float, default=0.8)
    ap.add_argument("--tsvad-iters", type=int, default=1, help="profile head passes; later ones take the "
                    "profiles from the solo speech the previous pass found")
    ap.add_argument("--tsvad", default="", help="ProfileHead checkpoint (train_tsvad.py): after linking, the "
                    "activity of A and B from the encoder features and the two recording profiles")
    ap.add_argument("--render-part", choices=("all", "sources", "finish"), default="all",
                    help="with --aug-root: the whole recording chain, only what acts on each voice (room, "
                    "EQ, bleed, telephone band, compression) or only what acts on the mixture (noise, music, "
                    "gain, clipping, codec)")
    a = ap.parse_args()
    lp = json.loads(a.link_params)
    link = build_memory if a.linker == "memory" else (lambda ws: link_windows(ws, **lp))
    if a.split == "test" and not a.final:
        raise SystemExit("the test split is read only with --final")
    _, val, test = splits(a.data, split_file=a.split_file or None)
    _train = None
    if a.split == "train":  # the train/val gap of activity; conversations spread over train
        _train, _, _ = splits(a.data, split_file=a.split_file or None)
        _train = _train[:: max(1, len(_train) // a.conversations)]
    pairs = (_train if a.split == "train" else val if a.split == "val" else test)[: a.conversations]
    model = load(a.model, a.device)
    tsvad = None
    if a.tsvad:
        from m2s_model import ProfileHead

        tsvad = []  # several heads (comma-separated): their probabilities are averaged
        for path in a.tsvad.split(","):
            ck = torch.load(path, map_location="cpu")
            hd = ProfileHead(ck["cfg"]["hdim"], ck["cfg"]["zdim"], ck["cfg"]["layers"], ck["cfg"].get("sim", False),
                             ck["cfg"].get("scales"))
            hd.load_state_dict(ck["head"])
            if "enc" in ck:  # the encoder was trained with this head: its frames for the head, the
                # original one for the identity pass and linking (its activity head was not trained)
                import copy

                hd.enc = copy.deepcopy(model.enc)
                hd.enc._speaker = model.enc._speaker
                hd.enc.load_state_dict(ck["enc"])
                hd.enc.eval()
            tsvad.append(hd.to(a.device).eval())
    render = None
    if a.aug_root:
        sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
        from render import Render

        from kezek.augment import DomainBank

        r = a.aug_root
        render = Render(DomainBank.from_dirs(
            [f"{r}/RIRS_NOISES/simulated_rirs", f"{r}/RIRS_NOISES/real_rirs_isotropic_noises"],
            [f"{r}/musan/noise"], [f"{r}/musan/music"]))
    sfx = "_rec" if render is not None else ""
    variant = ("_pack" if os.environ.get("KEZEK_PACK_SOLO") == "1" else "") + ("_ecapaft" if os.environ.get("KEZEK_ECAPA") else "")
    run = tracker.init(f"s1_full_{a.split}{sfx}_{os.path.basename(os.path.dirname(a.model))}{variant}{a.tag}",
                       {**vars(a), "pack_solo": os.environ.get("KEZEK_PACK_SOLO", "0"), "ecapa": os.environ.get("KEZEK_ECAPA", "")},
                       ["kezek-v3", "stage1", "full-recording", a.split], len(pairs))
    per_conv, hours, t_ident, h_ident, win_rows, win_runs = [], 0.0, 0.0, 0.0, [], []
    for k, pair in enumerate(pairs):
        c = _Conversation(pair)
        nfr = min(c.frames, int(a.max_minutes * 6000))
        sa, sb = c.audio(0, 0, nfr), c.audio(1, 0, nfr)
        act10 = np.stack([c.act[0][:nfr], c.act[1][:nfr]])
        if render is not None:
            rng = np.random.default_rng([30_000, k])
            rec = render.sample(rng)
            mask = np.repeat(act10[0] | act10[1], FRAME)[: len(sa)]
            # recordings over 30 min are rendered in 20 min pieces with the same chain (a 126 min
            # recording needs more than 15 GB in one piece); shorter ones in one piece as before
            step = len(sa) if len(sa) <= 30 * 60 * SR else 20 * 60 * SR
            x = np.concatenate([render_part(render, rec, sa[i:i + step], sb[i:i + step], mask[i:i + step], rng, a.render_part)
                                for i in range(0, len(sa), step)]).astype(np.float32)
        else:
            x = sa + sb
        x = (x * np.float32(0.9 / max(float(np.abs(x).max()), 1e-4))).astype(np.float32)
        cf = f"{a.cache}/{pair.conversation_id}{'_rec' if render is not None else ''}.pt" if a.cache else ""
        if cf and os.path.exists(cf):
            windows = torch.load(cf)
        else:
            t0 = time.time()
            windows = identity_pass(model, x, a.device, win_s=a.win_s, hop_s=a.hop_s)
            t_ident += time.time() - t0
            h_ident += len(x) / SR / 3600
            if cf:
                os.makedirs(a.cache, exist_ok=True)
                torch.save(windows, cf)
        if a.decode != "marginal":
            decode_windows(windows, a.decode, a.overlap_thr)
            a.reembed = a.reembed or "pred"
        if a.reembed:
            nf0 = len(x) // 320 + 1
            reembed(windows, x, model.enc._speaker[0], a.device, frames_20ms(torch.from_numpy(act10), nf0).bool().numpy(),
                    a.reembed, a.solo_thr)
        cents = link(windows)
        if a.relink:
            relink_segments(windows, x, model.enc._speaker[0], a.device, **json.loads(a.relink))
        if tsvad is not None and cents[0] is not None and cents[1] is not None:
            p_act = tsvad_pass(model, tsvad, x, windows, cents, a.device, a.tsvad_iters) > a.threshold
        else:
            p_act = memory_activity(windows, len(x), a.combine) > a.threshold
        nf = p_act.shape[1]
        bc10 = np.stack([backchannel_frames(act10[0], act10[1]), backchannel_frames(act10[1], act10[0])])
        act = frames_20ms(torch.from_numpy(act10), nf).bool().numpy()
        bc = frames_20ms(torch.from_numpy(bc10), nf).bool().numpy()
        unc = np.zeros(nf, bool)
        for w in windows:
            if not w["confident"]:
                unc[w["start"] // 320:w["end"] // 320] = True
        r = score_conversation(p_act, act, bc, unc)
        refs = references((sa, sb), act, model.enc._speaker[0], a.device)
        rows, runs = window_metrics(windows, p_act, act, refs)
        for row in rows:
            row["conv"] = pair.conversation_id
        win_rows += rows
        win_runs += runs
        r.update({"swap16": sum(bool(x.get("swap")) for x in rows), "clear16": sum(x["clear"] for x in rows)})
        r.update({"conv": pair.conversation_id, "minutes": nfr / 6000,
                  "uncertain_windows": int(sum(not w["confident"] for w in windows)), "windows": len(windows)})
        per_conv.append(r)
        hours += nfr / 360_000
        print(json.dumps(r), flush=True)
        tracker.log(run, {f"progress_{a.split}{sfx}/conversations_without_swaps": float(np.mean([p["swap_windows"] == 0 for p in per_conv])),
                          f"progress_{a.split}{sfx}/swap_windows": sum(p["swap_windows"] for p in per_conv)}, k + 1)
    tp, fp, fn = (sum(p[x] for p in per_conv) for x in ("tp", "fp", "fn"))
    prec, rec = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    sw = sum(p["swap_windows"] for p in per_conv)
    tpc, fpc, fnc = (sum(p[x] for p in per_conv) for x in ("tp_c", "fp_c", "fn_c"))
    pc, rc_ = tpc / max(tpc + fpc, 1e-9), tpc / max(tpc + fnc, 1e-9)
    fps = max(fp, 1e-9)
    out = {
        "act_f1_collar200": 2 * pc * rc_ / max(pc + rc_, 1e-9),
        "fp_share_edge": sum(p["fp_edge"] for p in per_conv) / fps,
        "fp_share_other_talks": sum(p["fp_other"] for p in per_conv) / fps,
        "fp_share_silence": sum(p["fp_silence"] for p in per_conv) / fps,
        "act_precision": prec, "act_recall": rec, "act_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "bc_recall": sum(p["bc_hit"] for p in per_conv) / max(sum(p["bc_n"] for p in per_conv), 1),
        "bc_runs": sum(p["bc_n"] for p in per_conv),
        "bc_recall_any_slot": sum(p["bc_hit_any"] for p in per_conv) / max(sum(p["bc_n"] for p in per_conv), 1),
        "swap_windows": sw, "swap_windows_per_hour": sw / max(hours, 1e-9),
        "swap_windows_unflagged": sw - sum(p["swap_windows_flagged"] for p in per_conv),
        "swap_window_rate": sw / max(sum(p["two_speaker_windows"] for p in per_conv), 1),
        "conversations_without_swaps": float(np.mean([p["swap_windows"] == 0 for p in per_conv])),
        "uncertain_window_rate": sum(p["uncertain_windows"] for p in per_conv) / max(sum(p["windows"] for p in per_conv), 1),
        "conversations": len(per_conv), "hours": hours,
    }
    clear = [x for x in win_rows if x["clear"]]
    sw16 = [x for x in clear if x["swap"]]
    unclear = [x for x in win_rows if "wrong_profile" in x]
    ids = [x["id_ok"] for x in clear if "id_ok" in x]
    out.update({
        "windows16": len(win_rows), "clear_windows16": len(clear), "unclear_windows16": len(win_rows) - len(clear),
        "swap_windows16": len(sw16), "swap_rate16": len(sw16) / max(len(clear), 1),
        "conversations_without_swaps16": float(np.mean([p["swap16"] == 0 for p in per_conv])),
        "unclear_wrong_profile_rate": float(np.mean([x["wrong_profile"] for x in unclear])) if unclear else float("nan"),
        "id_accuracy": float(np.mean(ids)) if ids else float("nan"),
        "swap16_edge_share": float(np.mean([x["edge"] for x in sw16])) if sw16 else float("nan"),
        "swap16_run_mean": float(np.mean(win_runs)) if win_runs else 0.0,
        "swap16_run_max": int(max(win_runs)) if win_runs else 0,
        "swap16_pred_solo_min_median": float(np.median([x["pred_solo_min"] for x in sw16])) if sw16 else float("nan"),
        "ok16_pred_solo_min_median": float(np.median([x["pred_solo_min"] for x in clear if not x["swap"]])) if clear else float("nan"),
        "swap16_ref_solo_min_median": float(np.median([x["ref_solo_min"] for x in sw16])) if sw16 else float("nan"),
        "swap_rate16_pred_solo_lt2": (lambda c: float(np.mean([x["swap"] for x in c])) if c else float("nan"))(
            [x for x in clear if x["pred_solo_min"] < 2.0]),
        "swap_rate16_pred_solo_ge2": (lambda c: float(np.mean([x["swap"] for x in c])) if c else float("nan"))(
            [x for x in clear if x["pred_solo_min"] >= 2.0]),
        "ident_s_per_audio_hour": t_ident / h_ident if h_ident else float("nan"),
    })
    if a.real_clip:
        import soundfile as sf
        from scipy.signal import resample_poly

        y, rate = sf.read(a.real_clip, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(1)
        if rate != SR:
            g = np.gcd(rate, SR)
            y = resample_poly(y, SR // g, rate // g).astype(np.float32)
        y = y * np.float32(0.9 / max(float(np.abs(y).max()), 1e-6))
        windows = identity_pass(model, y, a.device, win_s=a.win_s, hop_s=a.hop_s)
        link(windows)
        p_act = memory_activity(windows, len(y)) > 0.5
        n = p_act.shape[1]
        e = np.log10(np.mean(np.pad(y, (0, n * 320 - len(y)))[: n * 320].reshape(n, 320) ** 2, 1) + 1e-12) * 10
        speech = e > e.max() - 40
        share = p_act[:, speech].mean(1)
        out.update({"real_dominant_share": float(share.max()), "real_other_share": float(share.min())})
    json.dump({"summary": out, "per_conversation": per_conv, "windows": win_rows, "swap_runs": win_runs, "args": vars(a)}, open(a.out, "w"), indent=1)
    print("SUMMARY", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}), flush=True)
    tracker.log(run, {f"full_{a.split}{sfx}/{k}": v for k, v in out.items() if isinstance(v, (int, float))}, len(pairs))
    tracker.finish(run)


if __name__ == "__main__":
    main()
