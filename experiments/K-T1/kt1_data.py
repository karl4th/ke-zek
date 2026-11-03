"""K-T1 data: LibriSpeech speakers -> simulated conversation windows with domain augmentation.

Window i is a deterministic function of (seed, i), so training resumes without repeats and the
validation set is fixed.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

from kezek.augment import AugmentConfig, DomainBank, sample_chain
from kezek.mixing import ConversationConfig, simulate_conversation

SR = 16000
MIN_S = 7.0


def build_index(roots: list[str], cache: str | None = None) -> dict[str, list[str]]:
    """speaker id -> utterances of at least 7 s; speakers with fewer than 3 such are dropped
    (two utterances for the turns, a third for the enrollment)."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    files = sorted(f for r in roots for f in glob.glob(f"{r}/*/*/*.flac"))
    with ThreadPoolExecutor(32) as ex:
        frames = list(ex.map(lambda f: sf.info(f).frames, files))
    index: dict[str, list[str]] = {}
    for f, n in zip(files, frames, strict=True):
        if n >= MIN_S * SR:
            index.setdefault(os.path.basename(os.path.dirname(os.path.dirname(f))), []).append(f)
    index = {k: v for k, v in index.items() if len(v) >= 3}
    if cache:
        json.dump(index, open(cache, "w"))
    return index


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _list_hash(paths: list[str], roots: list[str]) -> str:
    """Hash of a file list, independent of where the data lives on disk."""
    rel = []
    for p in paths:
        r = next(
            (
                os.path.relpath(p, root)
                for root in roots
                if os.path.abspath(p).startswith(os.path.abspath(root))
            ),
            p,
        )
        rel.append(r.replace(os.sep, "/"))
    return hashlib.sha256("\n".join(sorted(rel)).encode()).hexdigest()


def manifest(args: dict, bank: DomainBank, train_idx: dict, val_idx: dict, init: str) -> dict:
    """What a run was made of: code, start checkpoint, data and background lists (as hashes)."""
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__),
            ).stdout.strip()
        )
    except Exception:
        commit, dirty = None, None
    utts = lambda idx: [u for v in idx.values() for u in v]  # noqa: E731
    return {
        "commit": commit,
        "dirty": dirty,
        "args": args,
        "init_sha256": sha256_file(init),
        "train": {
            "speakers": len(train_idx),
            "utterances": len(utts(train_idx)),
            "list_sha256": _list_hash(utts(train_idx), args["train_roots"]),
        },
        "val": {
            "speakers": len(val_idx),
            "utterances": len(utts(val_idx)),
            "list_sha256": _list_hash(utts(val_idx), [args["val_root"]]),
        },
        "rirs": {"n": len(bank.rirs), "list_sha256": _list_hash(bank.rirs, args["rir_dirs"])},
        "noises": {
            "n": len(bank.noises),
            "list_sha256": _list_hash(bank.noises, args["noise_dirs"]),
        },
        "music": {"n": len(bank.music), "list_sha256": _list_hash(bank.music, args["music_dirs"])},
        "conversation": vars_of(ConversationConfig()),
        "augment": vars_of(AugmentConfig()),
    }


def vars_of(cfg) -> dict:
    from dataclasses import asdict

    return asdict(cfg)


def _read(path: str) -> np.ndarray:
    return sf.read(path, dtype="float32")[0]


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        index,
        bank: DomainBank | None,
        n: int,
        seed: int,
        conv: ConversationConfig | None = None,
        aug: AugmentConfig | None = None,
    ):
        self.index, self.names = index, sorted(index)
        self.bank, self.n, self.seed = bank, n, seed
        self.conv, self.aug = conv or ConversationConfig(), aug or AugmentConfig()

    def __len__(self):
        return self.n

    def episode(self, i: int, keep_full: bool = False):
        rng = np.random.default_rng([self.seed, i])
        a, b = rng.choice(len(self.names), 2, replace=False)
        ua, ub = self.index[self.names[a]], self.index[self.names[b]]
        ia, ib = rng.choice(len(ua), 3, replace=False), rng.choice(len(ub), 3, replace=False)
        chain = sample_chain(self.bank, rng, self.aug)
        return simulate_conversation(
            [_read(ua[ia[0]]), _read(ua[ia[1]])],
            [_read(ub[ib[0]]), _read(ub[ib[1]])],
            _read(ua[ia[2]]),
            _read(ub[ib[2]]),
            rng,
            chain,
            self.conv,
            keep_full=keep_full,
        )

    def __getitem__(self, i):
        e = self.episode(i)
        return {
            "mix": e.mixture,
            "clean": e.target,
            "other": e.other,
            "act_t": e.act_t,
            "act_o": e.act_o,
            "disputed": e.disputed,
            "silent": ~e.act_t & ~e.disputed,
            "l_ref": np.float32(e.l_ref),
            "enr_t": e.enrollment,
            "enr_o": e.enrollment_other,
            "clean_window": e.clean,
            "sir_db": np.float32(e.sir_db),
            "layout": e.layout,
        }


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    """Enrollments differ in length (6-10 s): cut to the shortest in the batch."""
    out = {}
    for k in items[0]:
        if k in ("enr_t", "enr_o"):
            n = min(len(it[k]) for it in items)
            out[k] = torch.from_numpy(np.stack([it[k][:n] for it in items]))
        else:
            out[k] = torch.as_tensor(np.stack([np.asarray(it[k]) for it in items]))
    return out
