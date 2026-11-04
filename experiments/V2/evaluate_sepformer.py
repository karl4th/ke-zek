"""Evaluate a public two-speaker separator followed by voice-sample matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from train import git_state, select_splits, short_overlap_turn_score

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows

SEPARATOR_REPO = "speechbrain/sepformer-wsj02mix"
SPEAKER_REPO = "speechbrain/spkrec-ecapa-voxceleb"


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype("float32")


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio[:length]


def _voice_score(classifier, enrollment: np.ndarray, source: np.ndarray, device: str) -> float:
    if float(np.sqrt(np.mean(source**2))) < 1e-4:
        return -1.0

    def embed(audio: np.ndarray) -> torch.Tensor:
        rms = max(float(np.sqrt(np.mean(audio**2))), 1e-5)
        waveform = torch.from_numpy((audio * (0.1 / rms)).astype("float32"))[None].to(device)
        return classifier.encode_batch(waveform).flatten()

    return float(F.cosine_similarity(embed(enrollment), embed(source), dim=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=6.0)
    parser.add_argument("--run-name", default="v2-sepformer-ecapa-baseline")
    parser.add_argument("--separator-repo", default=SEPARATOR_REPO)
    parser.add_argument("--speaker-repo", default=SPEAKER_REPO)
    parser.add_argument("--separator-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-tracker", action="store_true")
    args = parser.parse_args()
    if args.examples < 1 or args.enrollment_seconds <= 0:
        parser.error("Examples and enrollment duration must be positive")
    if not args.no_tracker and (
        not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL")
    ):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.inference.separation import SepformerSeparation

    api = HfApi()
    separator_revision = api.model_info(args.separator_repo).sha
    speaker_revision = api.model_info(args.speaker_repo).sha
    root = args.out.parent
    root.mkdir(parents=True, exist_ok=True)
    separator = SepformerSeparation.from_hparams(
        source=args.separator_repo,
        savedir=str(root / "pretrained" / args.separator_repo.replace("/", "_")),
        run_opts={"device": args.device},
    )
    checkpoint_sha256 = None
    if args.separator_checkpoint:
        with args.separator_checkpoint.open("rb") as source:
            checkpoint_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        state = torch.load(args.separator_checkpoint, map_location="cpu", weights_only=True)
        if state["source_repo"] != args.separator_repo:
            raise ValueError("Checkpoint separator repository does not match")
        separator.mods.load_state_dict(state["model"])
        separator_revision = state["source_revision"]
    classifier = EncoderClassifier.from_hparams(
        source=args.speaker_repo,
        savedir=str(root / "pretrained" / args.speaker_repo.replace("/", "_")),
        run_opts={"device": args.device},
    )
    separator_rate = int(separator.hparams.sample_rate)
    _, val_pairs, _ = select_splits(scan_conversations(args.data_root), args.split_manifest, 42)
    dataset = OpenYapWindows(
        val_pairs, args.examples, seed=10_042,
        enrollment_seconds=args.enrollment_seconds,
    )
    commit, dirty = git_state()
    config = {
        "commit": commit,
        "dirty": dirty,
        "separator": args.separator_repo,
        "separator_revision": separator_revision,
        "separator_rate": separator_rate,
        "checkpoint_sha256": checkpoint_sha256,
        "speaker_encoder": args.speaker_repo,
        "speaker_revision": speaker_revision,
        "split_manifest": str(args.split_manifest),
        "examples": args.examples,
        "enrollment_seconds": args.enrollment_seconds,
        "val_eligible_conversations": len(dataset.pairs),
    }
    if dirty:
        raise RuntimeError("Evaluation must use a clean commit")
    tracker = None
    if not args.no_tracker:
        import airon

        tracker = airon.init(
            project="kezek",
            experiment="K-V2 recipe search on Open Yap",
            name=args.run_name,
            config=config,
        )
    scores: list[dict[str, float]] = []
    short_scores: list[float] = []
    selected_correct = []
    clean_selected_correct = []
    reverse_scores = []
    oracle_scores = []
    deltas = []
    demo = root / f"{args.run_name}-audio"
    demo.mkdir(exist_ok=True)
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                mixture = sample["mixture"].numpy()
                target = sample["target"].numpy()
                other = sample["other"].numpy()
                separator_input = _resample(mixture, 16_000, separator_rate)
                separated = separator.separate_batch(
                    torch.from_numpy(separator_input)[None].to(args.device)
                )[0].cpu().numpy()
                sources = [
                    _fit_length(_resample(separated[:, side], separator_rate, 16_000), len(mixture))
                    for side in range(2)
                ]
                reconstruction = sources[0] + sources[1]
                gain = float(
                    np.dot(mixture, reconstruction)
                    / max(np.dot(reconstruction, reconstruction), 1e-8)
                )
                sources = [source * gain for source in sources]
                enrollment = sample["enrollment"].numpy()
                if float(np.sqrt(np.mean(target**2))) >= 1e-3:
                    clean_scores = [
                        _voice_score(classifier, enrollment, clean_source, args.device)
                        for clean_source in (target, other)
                    ]
                    clean_selected_correct.append(float(clean_scores[0] > clean_scores[1]))
                scores_by_source = [
                    _voice_score(classifier, enrollment, source, args.device) for source in sources
                ]
                selected = int(np.argmax(scores_by_source))
                estimate = sources[selected]
                opposite_enrollment = sample["other_enrollment"].numpy()
                reverse_selected = int(np.argmax([
                    _voice_score(classifier, opposite_enrollment, source, args.device)
                    for source in sources
                ]))
                reverse = sources[reverse_selected]
                source_metrics = [
                    score_extraction(source, target, other, 16_000) for source in sources
                ]
                oracle = int(np.argmax([metric.sisdr for metric in source_metrics]))
                selected_correct.append(float(selected == oracle))
                if np.isfinite(source_metrics[oracle].sisdri_overlap):
                    oracle_scores.append(source_metrics[oracle].sisdri_overlap)
                scores.append(source_metrics[selected].as_dict())
                reverse_metric = score_extraction(reverse, other, target, 16_000)
                if np.isfinite(reverse_metric.sisdri_overlap):
                    reverse_scores.append(reverse_metric.sisdri_overlap)
                deltas.append(float(np.sqrt(np.mean((estimate - reverse) ** 2))) /
                              max(float(np.sqrt(np.mean(estimate**2))), 1e-6))
                short = short_overlap_turn_score(sample, estimate, target, other)
                if short is not None:
                    short_scores.append(short[0])
                if index < 4:
                    for name, audio in (
                        ("mixture", mixture),
                        ("target", target),
                        ("estimate", estimate),
                    ):
                        sf.write(demo / f"{index:03d}_{name}.wav", audio, 16_000, subtype="FLOAT")
                if (index + 1) % 16 == 0:
                    print(f"evaluated {index + 1}/{len(dataset)}", flush=True)
        summary = {
            key: float(np.mean(values))
            for key in scores[0]
            if (values := [score[key] for score in scores if np.isfinite(score[key])])
        }
        summary.update(
            speaker_assignment_accuracy=float(np.mean(selected_correct)),
            clean_speaker_assignment_examples=float(len(clean_selected_correct)),
            oracle_sisdri_overlap=float(np.mean(oracle_scores)),
            reverse_sisdri_overlap=float(np.mean(reverse_scores)),
            enrollment_delta_ratio=float(np.mean(deltas)),
        )
        if clean_selected_correct:
            summary["clean_speaker_assignment_accuracy"] = float(
                np.mean(clean_selected_correct)
            )
        if short_scores:
            summary["short_overlap_turn_improvement_db"] = float(np.mean(short_scores))
            summary["short_overlap_turn_examples"] = float(len(short_scores))
        args.out.write_text(json.dumps({"config": config, "metrics": summary}, indent=2))
        print(json.dumps(summary, sort_keys=True), flush=True)
        if tracker is not None:
            tracker.log({f"val/{key}": value for key, value in summary.items()}, step=0)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
