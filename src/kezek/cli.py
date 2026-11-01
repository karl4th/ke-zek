"""Command-line interface.

kezek evaluate --ref ref.rttm --hyp hyp.rttm [--uem test.uem] [--collar 0.0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from kezek.annotation import end_time, read_rttm, read_uem, to_activity
from kezek.frames import SCORING_FRAME_RATE_HZ, interval_to_mask, num_frames
from kezek.metrics import DERResult, DuplexCounts, diarization_error, duplex_counts


def evaluate(
    ref_path: Path,
    hyp_path: Path,
    uem_path: Path | None = None,
    collar_s: float = 0.0,
    merge_gap_s: float = 0.25,
    frame_rate: float = SCORING_FRAME_RATE_HZ,
) -> dict:
    refs = read_rttm(ref_path)
    hyps = read_rttm(hyp_path)
    uems = read_uem(uem_path) if uem_path else {}
    files = sorted(uems) if uems else sorted(refs)

    per_file = {}
    der_total, duplex_total = DERResult(), DuplexCounts()
    for file_id in files:
        ref_segs = refs.get(file_id, [])
        hyp_segs = hyps.get(file_id, [])
        duration = max(end_time(ref_segs), end_time(hyp_segs))
        uem = None
        if file_id in uems:
            duration = max(duration, max(e for _, e in uems[file_id]))
            n = num_frames(duration, frame_rate)
            uem = np.zeros(n, dtype=bool)
            for s, e in uems[file_id]:
                uem |= interval_to_mask(s, e, n, frame_rate)
        ref, _ = to_activity(ref_segs, duration, frame_rate)
        hyp, _ = to_activity(hyp_segs, duration, frame_rate)
        if ref.shape[0] == 0:
            ref = np.zeros((1, hyp.shape[1]), dtype=bool)
        if hyp.shape[0] == 0:
            hyp = np.zeros((1, ref.shape[1]), dtype=bool)

        d = diarization_error(ref, hyp, frame_rate, uem=uem, collar_s=collar_s)
        x = duplex_counts(ref, hyp, frame_rate, merge_gap_s=merge_gap_s, uem=uem)
        der_total += d
        duplex_total += x
        per_file[file_id] = {**d.as_dict(), **x.summary()}

    return {
        "settings": {
            "collar_s": collar_s,
            "merge_gap_s": merge_gap_s,
            "frame_rate": frame_rate,
            "num_files": len(files),
            "missing_hypotheses": sorted(set(files) - set(hyps)),
        },
        "total": {**der_total.as_dict(), **duplex_total.summary()},
        "files": per_file,
    }


def _fmt(v: float, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'-':>6}" if pct else f"{'-':>7}"
    return f"{100 * v:6.2f}" if pct else f"{v:7.1f}"


def _print_table(report: dict) -> None:
    head = (
        f"{'file':<32} {'DER%':>6} {'miss%':>6} {'FA%':>6} {'conf%':>6} "
        f"{'bnd50':>7} {'bnd90':>7} {'leak%':>6} {'ovR%':>6} {'ovP%':>6}"
    )
    print(head)
    print("-" * len(head))
    rows = list(report["files"].items()) + [("TOTAL", report["total"])]
    for name, m in rows:
        print(
            f"{name[:32]:<32} {_fmt(m['der'], True)} {_fmt(m['missed'], True)} "
            f"{_fmt(m['false_alarm'], True)} {_fmt(m['confusion'], True)} "
            f"{_fmt(m['boundary_median_ms'])} {_fmt(m['boundary_p90_ms'])} "
            f"{_fmt(m['leakage'], True)} {_fmt(m['overlap_recall'], True)} "
            f"{_fmt(m['overlap_precision'], True)}"
        )
    missing = report["settings"]["missing_hypotheses"]
    if missing:
        print(f"\nwarning: no hypothesis for {len(missing)} file(s): {', '.join(missing[:5])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kezek", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="score a hypothesis RTTM against a reference RTTM")
    ev.add_argument("--ref", type=Path, required=True)
    ev.add_argument("--hyp", type=Path, required=True)
    ev.add_argument("--uem", type=Path, help="scored regions; also defines the file list")
    ev.add_argument("--collar", type=float, default=0.0, help="seconds (default: 0)")
    ev.add_argument(
        "--merge-gap", type=float, default=0.25, help="pause bridged into a turn (seconds)"
    )
    ev.add_argument("--json", type=Path, help="write the full report here")

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        report = evaluate(args.ref, args.hyp, args.uem, args.collar, args.merge_gap)
        _print_table(report)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
