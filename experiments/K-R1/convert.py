"""Stream one Open Yap archive part from stdin into 16 kHz training arrays.

    aws s3 cp s3://.../part-0001.tar - | python experiments/K-R1/convert.py --out /data/oy16 \
        --part part-0001.tar --sha256 <expected>

Per conversation it writes ``<out>/conv_<id>/``:

* ``a.npy``, ``b.npy``: int16 mono tracks at 16 kHz, sample aligned;
* ``act_a.npy``, ``act_b.npy``: 10 ms energy activity of each clean track (uint8);
* ``meta.json`` and the transcripts as delivered, plus ``stats.json`` (duration, overlap share,
  crosstalk estimate).

The archive never touches the disk. Its SHA-256 is computed on the stream and checked at the
end; a mismatch removes everything the part wrote and exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from kezek.mixing import active_frames

SR = 16_000
FRAME = 160


class HashingReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.hash = raw, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, b):
        n = self.raw.readinto(b)
        if n:
            self.hash.update(memoryview(b)[:n])
        return n


def to_16k(data: bytes) -> np.ndarray:
    x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if x.ndim != 1:
        raise ValueError("track is not mono")
    if rate != SR:
        g = math.gcd(rate, SR)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)
    return x


def crosstalk_db(x: np.ndarray, act_self: np.ndarray, act_other: np.ndarray) -> float:
    """Energy of a track while only the other person talks, relative to its own speech."""
    m_self = np.repeat(act_self & ~act_other, FRAME)[: len(x)]
    m_leak = np.repeat(~act_self & act_other, FRAME)[: len(x)]
    if m_self.sum() < SR or m_leak.sum() < SR:
        return float("nan")
    e_self = float(np.mean(np.float64(x[m_self]) ** 2))
    e_leak = float(np.mean(np.float64(x[m_leak]) ** 2))
    return 10 * math.log10((e_leak + 1e-12) / (e_self + 1e-12))


def write_conversation(out: Path, files: dict[str, bytes]) -> str:
    meta = json.loads(files["meta.json"])
    cid = str(meta["conversation_id"])
    a, b = to_16k(files["speaker_a.wav"]), to_16k(files["speaker_b.wav"])
    if len(a) != len(b):
        raise ValueError(f"{cid}: tracks are not aligned ({len(a)} vs {len(b)})")
    n = len(a) // FRAME * FRAME
    a, b = a[:n], b[:n]
    act_a, act_b = active_frames(a, SR), active_frames(b, SR)
    d = out / f"conv_{cid}"
    tmp = out / f".tmp_conv_{cid}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    peak = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-9)
    scale = 32767 / peak if peak > 1.0 else 32767.0
    np.save(tmp / "a.npy", np.round(a * scale).astype(np.int16))
    np.save(tmp / "b.npy", np.round(b * scale).astype(np.int16))
    np.save(tmp / "act_a.npy", act_a.astype(np.uint8))
    np.save(tmp / "act_b.npy", act_b.astype(np.uint8))
    for name, data in files.items():
        if name.endswith(".json"):
            (tmp / name).write_bytes(data)
    stats = {
        "seconds": n / SR,
        "active_a_s": float(act_a.sum()) * 0.01,
        "active_b_s": float(act_b.sum()) * 0.01,
        "overlap_s": float((act_a & act_b).sum()) * 0.01,
        "crosstalk_a_db": crosstalk_db(a, act_a, act_b),
        "crosstalk_b_db": crosstalk_db(b, act_b, act_a),
        "clipped_a": float(np.mean(np.abs(a) >= 0.999)),
        "clipped_b": float(np.mean(np.abs(b) >= 0.999)),
    }
    (tmp / "stats.json").write_text(json.dumps(stats))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    return cid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--sha256", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done_dir = out / ".parts_done"
    done_dir.mkdir(exist_ok=True)

    reader = HashingReader(sys.stdin.buffer.raw if hasattr(sys.stdin.buffer, "raw") else sys.stdin.buffer)
    stream = io.BufferedReader(reader, buffer_size=1 << 22)
    written: list[str] = []
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe member {member.name}")
            if len(parts) < 2 or parts[-1].startswith("._"):
                continue
            conv, name = parts[-2], parts[-1]
            pending.setdefault(conv, {})[name] = tar.extractfile(member).read()
            files = pending[conv]
            if {"meta.json", "speaker_a.wav", "speaker_b.wav"} <= files.keys() and all(
                f"speaker_{s}_transcript.json" in files for s in "ab"
            ):
                written.append(write_conversation(out, pending.pop(conv)))
                print(f"{a.part}: {written[-1]}", flush=True)
    while stream.read(1 << 22):
        pass
    for conv, files in pending.items():
        if {"meta.json", "speaker_a.wav", "speaker_b.wav"} <= files.keys():
            written.append(write_conversation(out, files))
        else:
            print(f"{a.part}: incomplete {conv}: {sorted(files)}", flush=True)
    digest = reader.hash.hexdigest()
    if digest != a.sha256:
        for cid in written:
            shutil.rmtree(out / f"conv_{cid}", ignore_errors=True)
        raise SystemExit(f"{a.part}: SHA-256 mismatch {digest} != {a.sha256}")
    (done_dir / a.part).write_text(json.dumps(written))
    print(f"{a.part}: done, {len(written)} conversations", flush=True)


if __name__ == "__main__":
    main()
