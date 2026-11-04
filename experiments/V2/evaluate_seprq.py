"""Evaluate a pretrained direct target extraction head on Open Yap windows."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from evaluate_sepformer import trim_by_energy
from train import git_state, select_splits, short_overlap_turn_score

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


def align_to_mixture(estimate: np.ndarray, mixture: np.ndarray) -> np.ndarray:
    """Undo a pretrained head's peak normalization using the observed mixture."""
    gain = float(np.dot(mixture, estimate)) / max(float(np.dot(estimate, estimate)), 1e-8)
    return estimate * np.clip(gain, -100.0, 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument(
        "--task", choices=("target-speaker-extraction", "personalized-extraction"),
        default="target-speaker-extraction",
    )
    parser.add_argument("--upstream", default="WavLM_BASE")
    parser.add_argument("--raw-output", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="v2-direct-pretrained-tse-val128")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.examples < 1 or args.enrollment_seconds <= 0:
        parser.error("Examples and enrollment duration must be positive")
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from seprq import SepRQPipeline

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Evaluation must use a clean commit")
    pipe = SepRQPipeline(args.task, upstream=args.upstream, device=args.device)
    pipe.eval()
    _, val_pairs, test_pairs = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    selected = val_pairs if args.split == "val" else test_pairs
    dataset = OpenYapWindows(
        selected, args.examples, seed=10_042 if args.split == "val" else 20_042,
        enrollment_seconds=args.enrollment_seconds,
    )
    config = {
        "commit": commit,
        "package_version": version("seprq"),
        "model_revision": HfApi().model_info("SepRQ/SepRQ").sha,
        "task": args.task,
        "upstream": args.upstream,
        "gain_alignment": not args.raw_output,
        "split_manifest": str(args.split_manifest),
        "split": args.split,
        "examples": args.examples,
        "enrollment_seconds": args.enrollment_seconds,
        "dirty": dirty,
    }
    import airon

    tracker = airon.init(
        project="kezek", experiment="K-V2 recipe search on Open Yap",
        name=args.run_name, config=config,
    )
    scores = []
    short_scores = []
    rows = []
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                mixture = sample["mixture"]
                enrollment = torch.from_numpy(
                    trim_by_energy(sample["enrollment"].numpy()).copy()
                )
                estimate = pipe(mixture, enrollment=enrollment)
                if estimate.ndim != 1:
                    raise ValueError("Direct extractor must produce one mono waveform")
                target = sample["target"].numpy()
                other = sample["other"].numpy()
                estimate = estimate.detach().float().cpu().numpy()
                estimate = np.pad(
                    estimate[: len(target)], (0, max(0, len(target) - len(estimate)))
                )
                if not args.raw_output:
                    estimate = align_to_mixture(estimate, mixture.numpy())
                score = score_extraction(estimate, target, other, 16_000)
                scores.append(score.as_dict())
                short = short_overlap_turn_score(sample, estimate, target, other)
                if short is not None:
                    short_scores.append(short[0])
                rows.append({
                    "conversation_id": sample["conversation_id"],
                    "target_side": sample["target_side"],
                    "sisdri_overlap": (
                        score.sisdri_overlap if np.isfinite(score.sisdri_overlap) else None
                    ),
                    "short_overlap_turn_improvement_db": short[0] if short else None,
                })
                if (index + 1) % 16 == 0:
                    print(f"evaluated {index + 1}/{len(dataset)}", flush=True)
        metrics = {
            key: float(np.mean(values))
            for key in scores[0]
            if (values := [item[key] for item in scores if np.isfinite(item[key])])
        }
        if short_scores:
            metrics["short_overlap_turn_improvement_db"] = float(np.mean(short_scores))
            metrics["short_overlap_turn_examples"] = float(len(short_scores))
        report = {"config": config, "metrics": metrics, "rows": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(metrics, sort_keys=True), flush=True)
        tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=0)
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
