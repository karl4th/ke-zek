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


def trim_by_energy(audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Concatenate speech-energy frames, preserving the original if too little remains."""
    frame = sample_rate // 50
    count = len(audio) // frame
    if count < 1:
        return audio
    frames = audio[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    threshold = max(float(np.max(rms)) * 0.03, 1e-3)
    selected = frames[rms >= threshold].reshape(-1)
    return selected if len(selected) >= sample_rate // 2 else audio


def _speaker_embedding(
    classifier, audio: np.ndarray, device: str,
) -> torch.Tensor | None:
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-4:
        return None
    waveform = torch.from_numpy((audio * (0.1 / rms)).astype("float32"))[None].to(device)
    return classifier.encode_batch(waveform).flatten()


def _embedding_score(first: torch.Tensor | None, second: torch.Tensor | None) -> float:
    if first is None or second is None:
        return -1.0
    return float(F.cosine_similarity(first, second, dim=0))


def feature_vector(
    enrollment: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
    first_rms: float, second_rms: float,
) -> torch.Tensor:
    """Signed pair features: swapping the sources reverses their sign."""
    enrollment, first, second = (
        F.normalize(value.float(), dim=0) for value in (enrollment, first, second)
    )
    difference = first - second
    cosine_difference = torch.dot(enrollment, difference)
    energy_difference = np.log(max(first_rms, 1e-6) / max(second_rms, 1e-6))
    return torch.cat((
        enrollment * difference,
        cosine_difference[None],
        difference.new_tensor([energy_difference]),
    )).cpu()


def _voice_score(classifier, enrollment: np.ndarray, source: np.ndarray, device: str) -> float:
    return _embedding_score(
        _speaker_embedding(classifier, enrollment, device),
        _speaker_embedding(classifier, source, device),
    )


def _windowed_voice_score(
    classifier, enrollment: np.ndarray, source: np.ndarray, device: str,
    window_seconds: float,
) -> float:
    if window_seconds <= 0:
        return _voice_score(classifier, enrollment, source, device)
    width = round(window_seconds * 16_000)
    if len(source) <= width:
        return _voice_score(classifier, enrollment, source, device)
    hop = max(1, width // 2)
    starts = list(range(0, len(source) - width + 1, hop))
    if starts[-1] != len(source) - width:
        starts.append(len(source) - width)
    return max(
        _voice_score(classifier, enrollment, source[start : start + width], device)
        for start in starts
    )


def _select_source(
    classifier, enrollment: np.ndarray, matching_sources: list[np.ndarray],
    sources: list[np.ndarray], device: str, speaker_window_seconds: float,
    selector_state: dict | None,
) -> int:
    def baseline() -> int:
        return int(np.argmax([
            _windowed_voice_score(
                classifier, enrollment, source, device, speaker_window_seconds,
            )
            for source in matching_sources
        ]))

    if selector_state is None or selector_state["method"] == "cosine":
        return baseline()
    voice = _speaker_embedding(classifier, enrollment, device)
    source_embeddings = [
        _speaker_embedding(classifier, source, device) for source in matching_sources
    ]
    if voice is None or any(item is None for item in source_embeddings):
        return baseline()
    first, second = source_embeddings
    assert first is not None and second is not None
    rms = [float(np.sqrt(np.mean(source**2))) for source in sources]
    features = feature_vector(voice, first, second, *rms)
    score = torch.dot(features / selector_state["scale"], selector_state["weights"])
    return int(score < 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--mixture-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=6.0)
    parser.add_argument("--speaker-window-seconds", type=float, default=0.0)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--run-name", default="v2-sepformer-ecapa-baseline")
    parser.add_argument("--separator-repo", default=SEPARATOR_REPO)
    parser.add_argument("--speaker-repo", default=SPEAKER_REPO)
    parser.add_argument("--separator-checkpoint", type=Path)
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-tracker", action="store_true")
    args = parser.parse_args()
    if args.examples < 1 or min(args.mixture_seconds, args.enrollment_seconds) <= 0:
        parser.error("Examples and window durations must be positive")
    if args.speaker_window_seconds < 0:
        parser.error("Speaker window duration cannot be negative")
    if args.selector_checkpoint and args.speaker_window_seconds:
        parser.error("A learned selector cannot use windowed speaker scoring")
    if args.selector_checkpoint and not args.trim_silence:
        parser.error("A learned selector requires energy-trimmed speaker audio")
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
        checksum = hashlib.sha256()
        with args.separator_checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        checkpoint_sha256 = checksum.hexdigest()
        state = torch.load(args.separator_checkpoint, map_location="cpu", weights_only=True)
        if state["source_repo"] != args.separator_repo:
            raise ValueError("Checkpoint separator repository does not match")
        separator.mods.load_state_dict(state["model"])
        separator_revision = state["source_revision"]
    selector_state = None
    selector_sha256 = None
    if args.selector_checkpoint:
        checksum = hashlib.sha256()
        with args.selector_checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        selector_sha256 = checksum.hexdigest()
        selector_state = torch.load(args.selector_checkpoint, map_location="cpu", weights_only=True)
        if selector_state["config"]["separator_checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("Selector was trained for another separator checkpoint")
    classifier = EncoderClassifier.from_hparams(
        source=args.speaker_repo,
        savedir=str(root / "pretrained" / args.speaker_repo.replace("/", "_")),
        run_opts={"device": args.device},
    )
    if selector_state and selector_state["config"]["speaker_revision"] != speaker_revision:
        raise ValueError("Selector was trained for another speaker encoder revision")
    if selector_state and (
        selector_state["config"]["enrollment_seconds"] != args.enrollment_seconds
    ):
        raise ValueError("Selector was trained for another enrollment duration")
    separator_rate = int(separator.hparams.sample_rate)
    _, val_pairs, test_pairs = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    selected_pairs = val_pairs if args.split == "val" else test_pairs
    dataset = OpenYapWindows(
        selected_pairs, args.examples, seed=10_042 if args.split == "val" else 20_042,
        mixture_seconds=args.mixture_seconds,
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
        "selector_checkpoint_sha256": selector_sha256,
        "selector_method": selector_state["method"] if selector_state else "cosine",
        "speaker_encoder": args.speaker_repo,
        "speaker_revision": speaker_revision,
        "split_manifest": str(args.split_manifest),
        "split": args.split,
        "examples": args.examples,
        "mixture_seconds": args.mixture_seconds,
        "enrollment_seconds": args.enrollment_seconds,
        "speaker_window_seconds": args.speaker_window_seconds,
        "trim_silence": args.trim_silence,
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
    voiced_selected_correct = []
    overlap_selected_correct = []
    clean_selected_correct = []
    reverse_scores = []
    oracle_scores = []
    deltas = []
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
                if args.trim_silence:
                    enrollment = trim_by_energy(enrollment)
                    matching_sources = [trim_by_energy(source) for source in sources]
                else:
                    matching_sources = sources
                if float(np.sqrt(np.mean(target**2))) >= 1e-3:
                    clean_target = trim_by_energy(target) if args.trim_silence else target
                    clean_other = trim_by_energy(other) if args.trim_silence else other
                    clean_scores = [
                        _voice_score(classifier, enrollment, clean_source, args.device)
                        for clean_source in (clean_target, clean_other)
                    ]
                    clean_selected_correct.append(float(clean_scores[0] > clean_scores[1]))
                scores_by_source = [
                    _windowed_voice_score(
                        classifier, enrollment, source, args.device,
                        args.speaker_window_seconds,
                    )
                    for source in matching_sources
                ]
                selected = _select_source(
                    classifier, enrollment, matching_sources, sources, args.device,
                    args.speaker_window_seconds, selector_state,
                )
                estimate = sources[selected]
                opposite_enrollment = sample["other_enrollment"].numpy()
                if args.trim_silence:
                    opposite_enrollment = trim_by_energy(opposite_enrollment)
                reverse_selected = _select_source(
                    classifier, opposite_enrollment, matching_sources, sources, args.device,
                    args.speaker_window_seconds, selector_state,
                )
                reverse = sources[reverse_selected]
                source_metrics = [
                    score_extraction(source, target, other, 16_000) for source in sources
                ]
                oracle = int(np.argmax([metric.sisdr for metric in source_metrics]))
                selected_correct.append(float(selected == oracle))
                if float(np.sqrt(np.mean(target**2))) >= 1e-3:
                    voiced_selected_correct.append(float(selected == oracle))
                    if source_metrics[oracle].overlap_s >= 0.1:
                        overlap_selected_correct.append(float(selected == oracle))
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
                overlap_improvement = source_metrics[selected].sisdri_overlap
                rows.append({
                    "conversation_id": sample["conversation_id"],
                    "target_side": sample["target_side"],
                    "target_voiced": bool(float(np.sqrt(np.mean(target**2))) >= 1e-3),
                    "overlap_s": source_metrics[selected].overlap_s,
                    "speaker_selected_correct": bool(selected == oracle),
                    "speaker_score_margin": float(abs(scores_by_source[0] - scores_by_source[1])),
                    "sisdri_overlap": (
                        float(overlap_improvement)
                        if np.isfinite(overlap_improvement) else None
                    ),
                    "short_overlap_turn_improvement_db": short[0] if short else None,
                })
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
        if voiced_selected_correct:
            summary["voiced_speaker_assignment_accuracy"] = float(
                np.mean(voiced_selected_correct)
            )
            summary["voiced_speaker_assignment_examples"] = float(len(voiced_selected_correct))
        if overlap_selected_correct:
            summary["overlap_speaker_assignment_accuracy"] = float(
                np.mean(overlap_selected_correct)
            )
            summary["overlap_speaker_assignment_examples"] = float(len(overlap_selected_correct))
        if short_scores:
            summary["short_overlap_turn_improvement_db"] = float(np.mean(short_scores))
            summary["short_overlap_turn_examples"] = float(len(short_scores))
        report = {"config": config, "metrics": summary, "rows": rows}
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(summary, sort_keys=True), flush=True)
        if tracker is not None:
            tracker.log({f"val/{key}": value for key, value in summary.items()}, step=0)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
