"""Command-line interface.

Prepare a test item from a real two-channel conversation (one speaker per channel):

    kezek prepare conversation.wav --target-channel 0 --out items/conv01_ch0

writes ``mixture.wav`` (channels summed), ``target.wav``, ``interferer.wav`` (the test
segment) and ``enrollment.wav`` (target solo speech from outside the test segment).

Score extraction outputs (``estimate.wav`` in each item directory):

    kezek score items/* [--estimate estimate.wav] [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from kezek.enrollment import pick_enrollment
from kezek.metrics import regions, score_extraction


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return audio, sr


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), audio, sr, subtype="FLOAT")


def prepare(
    stereo_path: Path,
    out_dir: Path,
    target_channel: int,
    start_s: float = 60.0,
    end_s: float | None = None,
    enroll_s: float = 10.0,
) -> dict:
    audio, sr = _read(stereo_path)
    if audio.shape[1] != 2:
        raise ValueError(f"{stereo_path}: expected 2 channels, got {audio.shape[1]}")
    target, interferer = audio[:, target_channel], audio[:, 1 - target_channel]
    mixture = target + interferer
    end_s = len(mixture) / sr if end_s is None else end_s
    reg = regions(target, interferer, sr)
    enrollment, used = pick_enrollment(mixture, reg.solo, sr, enroll_s, exclude_s=(start_s, end_s))

    a, b = int(start_s * sr), int(end_s * sr)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "mixture.wav", mixture[a:b], sr)
    _write(out_dir / "target.wav", target[a:b], sr)
    _write(out_dir / "interferer.wav", interferer[a:b], sr)
    _write(out_dir / "enrollment.wav", enrollment, sr)
    meta = {
        "source": str(stereo_path),
        "target_channel": target_channel,
        "segment_s": [start_s, end_s],
        "enrollment_from_s": used,
        "sample_rate": sr,
    }
    (out_dir / "item.json").write_text(json.dumps(meta, indent=2))
    return meta


def score(item_dirs: list[Path], estimate_name: str = "estimate.wav") -> dict:
    per_item, missing = {}, []
    for d in item_dirs:
        if not (d / estimate_name).exists():
            missing.append(str(d))
            continue
        est, sr = _read(d / estimate_name)
        tgt, sr_t = _read(d / "target.wav")
        itf, _ = _read(d / "interferer.wav")
        mix, _ = _read(d / "mixture.wav")
        if sr != sr_t:
            raise ValueError(f"{d}: estimate is {sr} Hz, reference is {sr_t} Hz")
        n = len(tgt)
        est = np.pad(est[:n, 0], (0, max(0, n - len(est))))
        per_item[d.name] = score_extraction(est, tgt[:, 0], itf[:, 0], sr, mix[:, 0]).as_dict()

    keys = next(iter(per_item.values())).keys() if per_item else []
    mean = {k: float(np.nanmean([m[k] for m in per_item.values()])) for k in keys}
    return {"num_items": len(per_item), "missing": missing, "mean": mean, "items": per_item}


def _fmt(v: float) -> str:
    return f"{'-':>8}" if v is None or math.isnan(v) else f"{v:8.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kezek", description="Target speaker extraction tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="make a test item from a two-channel conversation")
    p.add_argument("stereo", type=Path)
    p.add_argument("--target-channel", type=int, choices=[0, 1], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=float, default=60.0, help="test segment start (s)")
    p.add_argument("--end", type=float, default=None, help="test segment end (s)")
    p.add_argument("--enroll", type=float, default=10.0, help="enrollment length (s)")

    s = sub.add_parser("score", help="score estimate.wav in item directories")
    s.add_argument("items", type=Path, nargs="+")
    s.add_argument("--estimate", default="estimate.wav")
    s.add_argument("--json", type=Path)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        meta = prepare(
            args.stereo, args.out, args.target_channel, args.start, args.end, args.enroll
        )
        print(json.dumps(meta, indent=2))
    elif args.command == "score":
        report = score(args.items, args.estimate)
        cols = ["sisdr_overlap", "sisdri_overlap", "sisdr_solo", "leakage", "residual_db"]
        print(f"{'item':<28}" + "".join(f"{c[:8]:>9}" for c in cols))
        for name, m in list(report["items"].items()) + [("MEAN", report["mean"])]:
            print(f"{name[:28]:<28}" + "".join(" " + _fmt(m.get(c, math.nan)) for c in cols))
        if report["missing"]:
            print(f"\nwarning: no {args.estimate} in {len(report['missing'])} item(s)")
        if args.json:
            args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
