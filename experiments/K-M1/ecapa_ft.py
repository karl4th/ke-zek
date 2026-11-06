"""Fine-tune the ECAPA speaker encoder of stage 1 for the condition it meets inside Kezek:
short solo pieces of one voice, joined end to end, from a recording chain (room, noise, codec,
a quiet second voice leaking in). The pretrained model already saw VoxCeleb; what it has not
seen is this condition, and on it the slot embeddings of two people in one recording are too
close (K-V3-S1: A/B link errors come from runs of windows decided by embeddings alone).

    python experiments/K-M1/ecapa_ft.py --prep --vox /data/vox --shards 10      # parquet -> int16
    python experiments/K-M1/ecapa_ft.py --vox /data/vox --aug-root /data/aug --out /data/runs/ecapa1

Data: VoxCeleb2 dev, 16 kHz parquet shards of HF ``priyammaz/VoxCeleb2`` (rows shuffled over
speakers, so a few shards cover nearly all 5,994). 200 speakers are held out for validation.
Loss: additive angular margin softmax over the training speakers (margin 0.2, scale 30).
Validation: equal error rate on fixed trials of rendered pieces (all, and pieces <= 2 s); step 0
is the pretrained encoder. Output: ``ecapa.pt`` (state of the embedding model) for
``KEZEK_ECAPA=... m2s_infer``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
import tracker  # noqa: E402

SR = 16_000
REPO, REV = "priyammaz/VoxCeleb2", "main"


def prep(vox: str, shards: int) -> None:
    """Download ``shards`` train parquet shards and write audio.bin (int16, all utterances end
    to end, written shard by shard) with index.json (offset, length, speaker)."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    os.makedirs(vox, exist_ok=True)
    index, off = [], 0
    with open(f"{vox}/audio.bin", "wb") as out:
        for k in range(shards):
            f = hf_hub_download(REPO, f"data/train-{k:05d}-of-00142.parquet", repo_type="dataset",
                                revision=REV, local_dir=f"{vox}/parquet")
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                t = pf.read_row_group(g, columns=["audio", "speaker_id"])
                for a, spk in zip(t.column("audio").to_pylist(), t.column("speaker_id").to_pylist(), strict=True):
                    x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
                    if sr != SR or x.ndim != 1 or len(x) < SR:
                        continue
                    out.write(x.tobytes())
                    index.append((off, len(x), spk))
                    off += len(x)
            os.remove(f)
            print(f"shard {k}: {len(index)} utterances, {off / SR / 3600:.1f} h", flush=True)
    json.dump(index, open(f"{vox}/index.json", "w"))
    print(f"PREP DONE {len(index)} utterances, {len({s for _, _, s in index})} speakers", flush=True)


class Pieces:
    """One sample: a rendered crop of one utterance, cut into 1-5 pieces totalling 0.5-6 s and
    joined end to end; sometimes a quieter voice of another speaker leaks in."""

    def __init__(self, vox: str, aug_root: str):
        self.audio = np.memmap(f"{vox}/audio.bin", np.int16, "r")
        self.index = json.load(open(f"{vox}/index.json"))
        self.render = None
        if aug_root:
            from render import Render

            from kezek.augment import DomainBank

            r = aug_root
            self.render = Render(DomainBank.from_dirs(
                [f"{r}/RIRS_NOISES/simulated_rirs", f"{r}/RIRS_NOISES/real_rirs_isotropic_noises"],
                [f"{r}/musan/noise"], [f"{r}/musan/music"]))

    def crop(self, i: int, rng, sec: float) -> np.ndarray:
        o, n, _ = self.index[i]
        L = min(n, int(sec * SR))
        s = int(rng.integers(0, n - L + 1))
        return self.audio[o + s:o + s + L].astype(np.float32) / 32768

    def sample(self, i: int, rng, total=(0.5, 6.0), leak_p=0.3, other: int | None = None) -> np.ndarray:
        a = self.crop(i, rng, rng.uniform(3, 10))
        b = np.zeros_like(a)
        if other is not None and rng.random() < leak_p:
            y = self.crop(other, rng, len(a) / SR)
            b[: len(y)] = y * 10 ** (rng.uniform(-25, -10) / 20)
        if self.render is not None:
            _, _, x = self.render.apply(self.render.sample(rng), a, b, np.abs(a) > 1e-3, rng)
        else:
            x = a + b
        want = min(int(rng.uniform(*total) * SR), len(x))
        k = int(rng.integers(1, 6))
        cuts = np.sort(rng.integers(0, len(x), size=2 * k)).reshape(k, 2)
        parts = [x[s:e] for s, e in cuts if e - s > 160]
        y = np.concatenate(parts) if parts else x
        y = y[:want] if len(y) >= want else x[:want]
        return (y * np.float32(0.9 / max(float(np.abs(y).max()), 1e-4))).astype(np.float32)


class TrainSet(torch.utils.data.IterableDataset):
    def __init__(self, pieces_args, utts: list[int], spk_of: list[int], seed: int):
        self.args, self.utts, self.spk_of, self.seed = pieces_args, utts, spk_of, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid, int(time.time())])
        p = Pieces(*self.args)
        while True:
            i = self.utts[int(rng.integers(len(self.utts)))]
            o = self.utts[int(rng.integers(len(self.utts)))]
            yield p.sample(i, rng, other=o), self.spk_of[i]


def collate(batch):
    L = max(len(x) for x, _ in batch)
    wav = torch.stack([F.pad(torch.from_numpy(x), (0, L - len(x))) for x, _ in batch])
    rel = torch.tensor([len(x) / L for x, _ in batch])
    return wav, rel, torch.tensor([y for _, y in batch])


class AAM(nn.Module):
    def __init__(self, dim: int, n: int, margin=0.2, scale=30.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, dim) * 0.01)
        self.m, self.s = margin, scale

    def forward(self, z, y):
        cos = F.linear(F.normalize(z), F.normalize(self.w)).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos.gather(1, y[:, None]))
        tgt = torch.cos(theta + self.m)
        logits = cos.scatter(1, y[:, None], tgt) * self.s
        return F.cross_entropy(logits, y)


def eer(scores: np.ndarray, same: np.ndarray) -> float:
    order = np.argsort(-scores)
    s = same[order]
    tp = np.cumsum(s) / max(s.sum(), 1)
    fp = np.cumsum(~s) / max((~s).sum(), 1)
    fnr = 1 - tp
    k = int(np.argmin(np.abs(fnr - fp)))
    return float((fnr[k] + fp[k]) / 2)


def make_trials(pieces: Pieces, utts_by_spk: dict, n=3000, seed=7):
    """Fixed validation trials: (wav1, wav2, same, short) over held-out speakers."""
    rng = np.random.default_rng(seed)
    spks = sorted(utts_by_spk)
    out = []
    for t in range(n):
        same = t % 2 == 0
        s1 = spks[int(rng.integers(len(spks)))]
        s2 = s1 if same else spks[int(rng.integers(len(spks)))]
        while not same and s2 == s1:
            s2 = spks[int(rng.integers(len(spks)))]
        u1 = utts_by_spk[s1][int(rng.integers(len(utts_by_spk[s1])))]
        u2 = utts_by_spk[s2][int(rng.integers(len(utts_by_spk[s2])))]
        short = t % 4 < 2
        total = (0.5, 2.0) if short else (2.0, 6.0)
        other = utts_by_spk[spks[int(rng.integers(len(spks)))]][0]
        out.append((pieces.sample(u1, rng, total, other=other), pieces.sample(u2, rng, total, other=other), same, short))
    return out


@torch.no_grad()
def validate(embed, trials, device, bs=64):
    zs = []
    flat = [w for a, b, _, _ in trials for w in (a, b)]
    for i in range(0, len(flat), bs):
        wav, rel, _ = collate([(x, 0) for x in flat[i:i + bs]])
        zs.append(F.normalize(embed(wav.to(device), rel.to(device)), dim=-1).cpu())
    z = torch.cat(zs).view(len(trials), 2, -1)
    sc = (z[:, 0] * z[:, 1]).sum(-1).numpy()
    same = np.array([t[2] for t in trials])
    short = np.array([t[3] for t in trials])
    return {"eer": eer(sc, same), "eer_short": eer(sc[short], same[short]), "eer_long": eer(sc[~short], same[~short])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vox", required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--aug-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-speakers", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default="", help="embedding model weights to continue from (ecapa_last.pt)")
    ap.add_argument("--start-step", type=int, default=0, help="step those weights were saved at")
    a = ap.parse_args()
    if a.prep:
        prep(a.vox, a.shards)
        return
    os.makedirs(a.out, exist_ok=True)
    index = json.load(open(f"{a.vox}/index.json"))
    spks = sorted({s for _, _, s in index})
    rng = np.random.default_rng(0)
    val_spk = set(rng.choice(spks, a.val_speakers, replace=False).tolist())
    train_spk = [s for s in spks if s not in val_spk]
    cls = {s: k for k, s in enumerate(train_spk)}
    spk_of = [cls.get(s, -1) for _, _, s in index]
    train_utts = [i for i, (_, _, s) in enumerate(index) if s in cls]
    val_by = {}
    for i, (_, _, s) in enumerate(index):
        if s in val_spk:
            val_by.setdefault(s, []).append(i)

    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained/ecapa",
                                         run_opts={"device": a.device})
    model = enc.mods.embedding_model
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(True)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=a.device))

    def embed(wav, rel):
        feats = enc.mods.compute_features(wav)
        feats = enc.mods.mean_var_norm(feats, rel)
        return model(feats, rel).squeeze(1)

    head = AAM(192, len(train_spk)).to(a.device)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": a.lr},
                             {"params": head.parameters(), "lr": a.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s + a.start_step, a.steps) / a.steps)))
    trials = make_trials(Pieces(a.vox, a.aug_root), val_by)
    run = tracker.init(f"ecapa_ft_{os.path.basename(a.out.rstrip('/'))}",
                       {**vars(a), "train_speakers": len(train_spk), "utterances": len(train_utts)},
                       ["kezek-v3", "stage1", "ecapa-ft", "voxceleb2"], a.steps)
    model.eval()
    v = validate(embed, trials, a.device)
    print(f"EVAL step {a.start_step}: {json.dumps(v)}", flush=True)
    tracker.log(run, {f"val/{k}": x for k, x in v.items()}, a.start_step)
    best = v["eer"]
    torch.save(model.state_dict(), f"{a.out}/ecapa.pt")
    dl = torch.utils.data.DataLoader(TrainSet((a.vox, a.aug_root), train_utts, spk_of, 1), batch_size=a.bs,
                                     num_workers=a.workers, collate_fn=collate)
    head_warm = a.start_step + 300  # train the new head alone first, so the encoder is not pushed by random weights
    t0 = time.time()
    for step, (wav, rel, y) in enumerate(dl, a.start_step + 1):
        model.train(step > head_warm)
        for p in model.parameters():
            p.requires_grad_(step > head_warm)
        z = embed(wav.to(a.device), rel.to(a.device))
        loss = head(z, y.to(a.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step:6d} loss {loss.item():.3f} {(step - a.start_step) / (time.time() - t0):.2f} it/s", flush=True)
            tracker.log(run, {"train/loss": loss.item()}, step)
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            v = validate(embed, trials, a.device)
            print(f"EVAL step {step}: {json.dumps(v)}", flush=True)
            tracker.log(run, {f"val/{k}": x for k, x in v.items()}, step)
            if v["eer"] < best:
                best = v["eer"]
                torch.save(model.state_dict(), f"{a.out}/ecapa.pt")
            torch.save(model.state_dict(), f"{a.out}/ecapa_last.pt")
            torch.save(head.state_dict(), f"{a.out}/head_last.pt")
        if step >= a.steps:
            break
    tracker.finish(run)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
