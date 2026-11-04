"""Evaluate voice-selected separation across a continuous held-out conversation excerpt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from evaluate_sepformer import (
    SEPARATOR_REPO,
    SPEAKER_REPO,
    _embedding_score,
    _fit_length,
    _resample,
    _speaker_embedding,
    trim_by_energy,
)
from train import git_state, select_splits, short_overlap_turn_score

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


def choose_sources(
    emissions: np.ndarray, continuity: np.ndarray, weight: float,
) -> list[int]:
    """Find the highest-scoring source path across adjacent chunks."""
    if emissions.ndim != 2 or emissions.shape[1] != 2:
        raise ValueError("Expected one pair of speaker scores per chunk")
    if continuity.shape != (len(emissions) - 1, 2, 2) or weight < 0:
        raise ValueError("Invalid continuity scores or weight")
    values = emissions[0].copy()
    back = []
    for index in range(1, len(emissions)):
        links = values[:, None] + weight * continuity[index - 1]
        previous = np.argmax(links, axis=0)
        values = emissions[index] + links[previous, np.arange(2)]
        back.append(previous)
    path = [int(np.argmax(values))]
    for previous in reversed(back):
        path.append(int(previous[path[-1]]))
    return path[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--separator-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="v2-long-separation-selection")
    parser.add_argument("--no-tracker", action="store_true")
    args = parser.parse_args()
    if min(args.examples, args.chunks) < 1 or min(args.chunk_seconds, args.enrollment_seconds) <= 0:
        parser.error("Examples, chunks, and durations must be positive")
    if not args.no_tracker and (
        not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL")
    ):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.inference.separation import SepformerSeparation

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Evaluation must use a clean commit")
    root = args.out.parent
    root.mkdir(parents=True, exist_ok=True)
    separator = SepformerSeparation.from_hparams(
        source=SEPARATOR_REPO,
        savedir=str(root / "pretrained" / SEPARATOR_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    checkpoint = torch.load(args.separator_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["source_repo"] != SEPARATOR_REPO:
        raise ValueError("Checkpoint separator repository does not match")
    separator.mods.load_state_dict(checkpoint["model"])
    classifier = EncoderClassifier.from_hparams(
        source=SPEAKER_REPO,
        savedir=str(root / "pretrained" / SPEAKER_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    checksum = hashlib.sha256()
    with args.separator_checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    _, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    dataset = OpenYapWindows(
        val_pairs, args.examples, seed=10_042,
        mixture_seconds=args.chunk_seconds * args.chunks,
        enrollment_seconds=args.enrollment_seconds,
    )
    config = {
        "commit": commit,
        "checkpoint_sha256": checksum.hexdigest(),
        "checkpoint_step": checkpoint["step"],
        "separator": SEPARATOR_REPO,
        "separator_revision": checkpoint["source_revision"],
        "speaker_encoder": SPEAKER_REPO,
        "speaker_revision": HfApi().model_info(SPEAKER_REPO).sha,
        "split_manifest": str(args.split_manifest),
        "examples": args.examples,
        "chunk_seconds": args.chunk_seconds,
        "chunks": args.chunks,
        "enrollment_seconds": args.enrollment_seconds,
        "weights": [0.0, 0.25, 0.5],
        "val_eligible_conversations": len(dataset.pairs),
    }
    tracker = None
    if not args.no_tracker:
        import airon

        tracker = airon.init(
            project="kezek", experiment="K-V2 recipe search on Open Yap",
            name=args.run_name, config=config,
        )
    results: dict[str, list[dict[str, float]]] = {
        str(weight): [] for weight in config["weights"]
    }
    rows = []
    demo = root / f"{args.run_name}-audio"
    demo.mkdir(exist_ok=True)
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                mixture = sample["mixture"].numpy()
                target = sample["target"].numpy()
                other = sample["other"].numpy()
                enrollment = trim_by_energy(sample["enrollment"].numpy())
                enrollment_embedding = _speaker_embedding(classifier, enrollment, args.device)
                chunk_frames = round(args.chunk_seconds * 16_000)
                sources_by_chunk = []
                embeddings_by_chunk = []
                emissions = np.empty((args.chunks, 2), dtype="float32")
                for chunk_index in range(args.chunks):
                    start = chunk_index * chunk_frames
                    stop = start + chunk_frames
                    chunk_mix = mixture[start:stop]
                    separator_rate = int(separator.hparams.sample_rate)
                    reduced = _resample(chunk_mix, 16_000, separator_rate)
                    separated = separator.separate_batch(
                        torch.from_numpy(reduced)[None].to(args.device)
                    )[0].cpu().numpy()
                    sources = [
                        _fit_length(
                            _resample(separated[:, side], separator_rate, 16_000),
                            len(chunk_mix),
                        )
                        for side in range(2)
                    ]
                    reconstruction = sources[0] + sources[1]
                    gain = float(
                        np.dot(chunk_mix, reconstruction)
                        / max(np.dot(reconstruction, reconstruction), 1e-8)
                    )
                    sources = [source * gain for source in sources]
                    embeddings = [
                        _speaker_embedding(classifier, trim_by_energy(source), args.device)
                        for source in sources
                    ]
                    emissions[chunk_index] = [
                        _embedding_score(enrollment_embedding, embedding)
                        for embedding in embeddings
                    ]
                    sources_by_chunk.append(sources)
                    embeddings_by_chunk.append(embeddings)
                continuity = np.empty((args.chunks - 1, 2, 2), dtype="float32")
                for chunk_index in range(1, args.chunks):
                    for before in range(2):
                        for after in range(2):
                            continuity[chunk_index - 1, before, after] = _embedding_score(
                                embeddings_by_chunk[chunk_index - 1][before],
                                embeddings_by_chunk[chunk_index][after],
                            )
                row = {
                    "conversation_id": sample["conversation_id"],
                    "target_side": sample["target_side"],
                }
                for weight in config["weights"]:
                    key = str(weight)
                    path = choose_sources(emissions, continuity, weight)
                    estimate = np.concatenate([
                        sources_by_chunk[chunk_index][side]
                        for chunk_index, side in enumerate(path)
                    ])
                    metrics = score_extraction(estimate, target, other, 16_000).as_dict()
                    short = short_overlap_turn_score(sample, estimate, target, other)
                    metrics["short_overlap_turn_improvement_db"] = (
                        short[0] if short else float("nan")
                    )
                    results[key].append(metrics)
                    row[key] = {
                        "path": path,
                        "sisdri_overlap": (
                            metrics["sisdri_overlap"]
                            if np.isfinite(metrics["sisdri_overlap"]) else None
                        ),
                    }
                    if index < 2 and weight in (0.0, 0.25):
                        sf.write(
                            demo / f"{index:03d}_weight{weight}_estimate.wav",
                            estimate, 16_000, subtype="FLOAT",
                        )
                if index < 2:
                    sf.write(demo / f"{index:03d}_mixture.wav", mixture, 16_000, subtype="FLOAT")
                    sf.write(demo / f"{index:03d}_target.wav", target, 16_000, subtype="FLOAT")
                rows.append(row)
                if (index + 1) % 8 == 0:
                    print(f"evaluated {index + 1}/{len(dataset)}", flush=True)
        summary = {
            key: {
                metric: float(np.mean(values))
                for metric in results[key][0]
                if (values := [
                    row[metric] for row in results[key] if np.isfinite(row[metric])
                ])
            }
            for key in results
        }
        report = {"config": config, "metrics": summary, "rows": rows}
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(summary, sort_keys=True), flush=True)
        if tracker is not None:
            tracker.log({
                f"val/weight_{weight}/{metric}": value
                for weight, metrics in summary.items() for metric, value in metrics.items()
            }, step=0)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
