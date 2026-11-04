"""Inspect paired-track sampling and reference activity before a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kezek.metrics import regions, score_extraction
from kezek.open_yap import scan_conversations, split_conversations
from kezek.open_yap_training import SAMPLE_RATE, OpenYapWindows
from kezek.vad import frames_to_samples


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def inspect_sample(sample: dict) -> dict[str, float]:
    target = sample["target"].numpy()
    other = sample["other"].numpy()
    mixture = sample["mixture"].numpy()
    enrollment = sample["enrollment"].numpy()
    reg = regions(target, other, SAMPLE_RATE)
    absent = frames_to_samples(reg.absent, SAMPLE_RATE, len(target))
    target_absent = float(np.mean(target[absent] ** 2)) if absent.any() else float("nan")
    other_absent = float(np.mean(other[absent] ** 2)) if absent.any() else float("nan")
    baseline = score_extraction(mixture, target, other, SAMPLE_RATE)
    return {
        "target_rms": _rms(target),
        "other_rms": _rms(other),
        "enrollment_rms": _rms(enrollment),
        "target_active_fraction": float(reg.overlap.mean() + reg.solo.mean()),
        "other_active_fraction": float(reg.overlap.mean() + reg.absent.mean()),
        "overlap_fraction": float(reg.overlap.mean()),
        "absent_fraction": float(reg.absent.mean()),
        "target_to_other_power_in_absent_db": float(
            10 * np.log10((target_absent + 1e-12) / (other_absent + 1e-12))
        ),
        "baseline_overlap_sisdri_db": baseline.sisdri_overlap,
        "baseline_absent_residual_db": baseline.residual_db,
        "mixture_sum_max_error": float(np.max(np.abs(mixture - target - other))),
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = samples[0]
    return {
        key: {
            "mean": float(np.mean(values)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "valid": len(values),
        }
        for key in keys
        if (values := [row[key] for row in samples if np.isfinite(row[key])])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    pairs = scan_conversations(args.data_root)
    train, val, test = split_conversations(pairs, seed=args.seed)
    report = {
        "conversations": {"train": len(train), "val": len(val), "test": len(test)},
        "splits": {},
    }
    for name, group, seed in (("train", train, args.seed), ("val", val, args.seed + 10_000)):
        dataset = OpenYapWindows(group, args.samples, seed)
        examples = [inspect_sample(dataset[index]) for index in range(args.samples)]
        report["splits"][name] = {"summary": summarize(examples), "examples": examples}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({name: data["summary"] for name, data in report["splits"].items()}))


if __name__ == "__main__":
    main()
