"""Inference on recordings of any length: 4 s windows, 1 s overlap, linear cross-fade (K-S mode).

No rescaling: a model trained with a scale-dependent loss (K-T1) is expected to output the
target at its level in the input.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_long(
    model,
    mixture: np.ndarray,
    d: torch.Tensor,
    sr: int = 16000,
    win_s: float = 4.0,
    overlap_s: float = 1.0,
    batch: int = 16,
) -> np.ndarray:
    """``mixture`` [T] float, ``d`` [1, D] enrollment embedding on the model's device."""
    win, hop = int(win_s * sr), int((win_s - overlap_s) * sr)
    x = np.asarray(mixture, dtype=np.float32)
    if len(x) <= win:
        dev = d.device
        return model(torch.from_numpy(x)[None].to(dev), d)[0].cpu().numpy()
    starts = list(range(0, len(x) - win, hop)) + [len(x) - win]
    fade = np.linspace(0.0, 1.0, win - hop, dtype=np.float32)
    out, wsum = np.zeros(len(x), np.float32), np.zeros(len(x), np.float32)
    for i in range(0, len(starts), batch):
        ss = starts[i : i + batch]
        seg = torch.from_numpy(np.stack([x[s : s + win] for s in ss])).to(d.device)
        y = model(seg, d.expand(len(ss), -1)).cpu().numpy()
        for s, yy in zip(ss, y, strict=True):
            w = np.ones(win, np.float32)
            if s > 0:
                w[: len(fade)] = fade
            if s + win < len(x):
                w[-len(fade) :] = fade[::-1]
            out[s : s + win] += w * yy
            wsum[s : s + win] += w
    return out / np.maximum(wsum, 1e-6)
