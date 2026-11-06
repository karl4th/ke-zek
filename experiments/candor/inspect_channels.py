"""How clean are CANDOR's per-speaker channels? For each conversation in an unpacked processed
archive: stereo mp3 (one participant per channel) decoded to 16 kHz, 20 ms frame levels, and

* active share per channel (frames within 30 dB of that channel's loud level);
* single talk: frames where one channel is active and the other is not;
* bleed: level of the quiet channel during the other's single talk, relative to the active one
  (median, dB), and the peak normalised cross-correlation of the two channels there (echo of the
  other voice gives a clear peak at a small lag; independent noise does not);
* identical channels and digital silence.

    python experiments/candor/inspect_channels.py /data/candor/unpacked
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 16_000
HOP = 320


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(path), "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).reshape(-1, 2).T.astype(np.float32) / 32768


def levels(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    return 10 * np.log10((x[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-10)


def xcorr_peak(a: np.ndarray, b: np.ndarray, max_lag: int = 8000) -> tuple[float, int]:
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    c = np.concatenate([c[-max_lag:], c[: max_lag + 1]]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    k = int(np.argmax(np.abs(c)))
    return float(abs(c[k])), k - max_lag


def main(root: str) -> None:
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        mp3 = d / "processed" / f"{d.name}.mp3"
        if not mp3.exists():
            continue
        x = decode(mp3)
        lv = np.stack([levels(x[0]), levels(x[1])])
        loud = np.percentile(lv, 95, axis=1)
        act = lv > (loud[:, None] - 30)
        row = {"conv": d.name, "minutes": round(x.shape[1] / SR / 60, 1),
               "identical": bool(np.allclose(x[0], x[1], atol=1e-4)),
               "digital_silence": [round(float((lv[c] < -90).mean()), 3) for c in (0, 1)],
               "active": [round(float(act[c].mean()), 3) for c in (0, 1)]}
        for c in (0, 1):
            single = act[c] & ~act[1 - c]
            row[f"single_{c}"] = round(float(single.mean()), 3)
            if single.sum() < 500:
                continue
            row[f"bleed_db_{c}"] = round(float(np.median(lv[1 - c, single] - lv[c, single])), 1)
            # longest stretch of single talk up to 60 s for the echo check
            f = np.flatnonzero(single)[:3000]
            a = np.concatenate([x[c, i * HOP:(i + 1) * HOP] for i in f])
            b = np.concatenate([x[1 - c, i * HOP:(i + 1) * HOP] for i in f])
            peak, lag = xcorr_peak(a, b)
            row[f"echo_corr_{c}"], row[f"echo_lag_ms_{c}"] = round(peak, 3), round(lag / SR * 1000)
        cm = d / "processed" / "channel_map.json"
        if cm.exists():
            row["channel_map"] = json.loads(cm.read_text())
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
