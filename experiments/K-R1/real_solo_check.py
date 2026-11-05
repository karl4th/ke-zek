"""Real-recording check without labels: on a stretch where only one person talks, a target
extractor should return its input. The first ``--enroll-s`` seconds of the clip are the voice
sample; on the rest, report SI-SDR(output, input), output level vs input and the share of 0.1 s
frames suppressed by more than 15 dB.

    python experiments/K-R1/real_solo_check.py --clip solo.wav --models a.pt b.pt ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from kezek.infer import extract_long
from kezek.metrics import si_sdr
from kezek.model import KezekTSE, SpeakerEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--enroll-s", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    x, sr = sf.read(a.clip, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    x = x * np.float32(0.9 / max(float(np.abs(x).max()), 1e-6))
    n_enr = int(a.enroll_s * 16000)
    enr, test = x[:n_enr], x[n_enr:]
    se = SpeakerEncoder(device=a.device)
    with torch.no_grad():
        d = se(torch.from_numpy(enr * np.float32(0.9 / max(float(np.abs(enr).max()), 1e-6)))[None].to(a.device))
    out = {}
    for path in a.models:
        m = KezekTSE().to(a.device).eval()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            y = extract_long(m, test, d, win_s=4.0, overlap_s=1.0)
        f = 1600
        n = len(test) // f
        ein = np.mean(test[: n * f].reshape(n, f) ** 2, 1) + 1e-12
        eout = np.mean(y[: n * f].reshape(n, f) ** 2, 1) + 1e-12
        speech = 10 * np.log10(ein) > 10 * np.log10(ein).max() - 40
        drop = 10 * np.log10(eout / ein)[speech]
        out[path] = {
            "solo_sisdr_db": round(si_sdr(y, test), 2),
            "level_db": round(float(10 * np.log10(eout[speech].sum() / ein[speech].sum())), 2),
            "suppressed_share": round(float(np.mean(drop < -15)), 3),
            "kept_share": round(float(np.mean(drop > -3)), 3),
        }
        print(path, json.dumps(out[path]), flush=True)


if __name__ == "__main__":
    main()
