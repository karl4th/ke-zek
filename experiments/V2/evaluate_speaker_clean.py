"""Compare voice-sample matchers on clean held-out Open Yap speaker tracks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from evaluate_sepformer import SPEAKER_REPO, _voice_score
from train import git_state, select_splits

from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


def trim_by_energy(audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Concatenate speech-energy frames, falling back to the original when too short."""
    frame = sample_rate // 50
    count = len(audio) // frame
    if count < 1:
        return audio
    frames = audio[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    threshold = max(float(np.max(rms)) * 0.03, 1e-3)
    selected = frames[rms >= threshold].reshape(-1)
    return selected if len(selected) >= sample_rate // 2 else audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speaker-repo", default=SPEAKER_REPO)
    parser.add_argument("--enrollment-selection", choices=("ratio", "power"), default="ratio")
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="v2-clean-speaker-matching")
    parser.add_argument("--no-tracker", action="store_true")
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("Examples must be positive")
    if not args.no_tracker and (
        not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL")
    ):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from speechbrain.inference.classifiers import EncoderClassifier

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Evaluation must use a clean commit")
    revision = HfApi().model_info(args.speaker_repo).sha
    model = EncoderClassifier.from_hparams(
        source=args.speaker_repo,
        savedir=str(args.out.parent / "pretrained" / args.speaker_repo.replace("/", "_")),
        run_opts={"device": args.device},
    )
    _, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    dataset = OpenYapWindows(
        val_pairs, args.examples, seed=10_042,
        enrollment_selection=args.enrollment_selection,
    )
    config = {
        "commit": commit,
        "speaker_repo": args.speaker_repo,
        "speaker_revision": revision,
        "enrollment_selection": args.enrollment_selection,
        "trim_silence": args.trim_silence,
        "examples": args.examples,
        "split_manifest": str(args.split_manifest),
    }
    tracker = None
    if not args.no_tracker:
        import airon

        tracker = airon.init(
            project="kezek",
            experiment="K-V2 recipe search on Open Yap",
            name=args.run_name,
            config=config,
        )
    correct = []
    margins = []
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            target = sample["target"].numpy()
            other = sample["other"].numpy()
            enrollment = sample["enrollment"].numpy()
            if float(np.sqrt(np.mean(target**2))) < 1e-3:
                continue
            if args.trim_silence:
                target = trim_by_energy(target)
                other = trim_by_energy(other)
                enrollment = trim_by_energy(enrollment)
            same = _voice_score(model, enrollment, target, args.device)
            different = _voice_score(model, enrollment, other, args.device)
            margins.append(same - different)
            correct.append(float(same > different))
        if not correct:
            raise RuntimeError("No voiced target examples were available")
        metrics = {
            "accuracy": float(np.mean(correct)),
            "mean_margin": float(np.mean(margins)),
            "examples": float(len(correct)),
        }
        args.out.write_text(json.dumps({"config": config, "metrics": metrics}, indent=2))
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if tracker is not None:
            tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=0)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
