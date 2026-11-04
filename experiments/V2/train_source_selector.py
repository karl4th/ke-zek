"""Fit a speaker-disjoint source selector on frozen separator and voice embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from evaluate_sepformer import (
    SEPARATOR_REPO,
    SPEAKER_REPO,
    _fit_length,
    _resample,
    _speaker_embedding,
    trim_by_energy,
)
from train import git_state, select_splits, short_overlap_turn_score

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


def feature_vector(enrollment: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
                   first_rms: float, second_rms: float) -> torch.Tensor:
    """Signed pair features: swapping the sources reverses their sign."""
    enrollment, first, second = (F.normalize(value.float(), dim=0)
                                 for value in (enrollment, first, second))
    difference = first - second
    cosine_difference = torch.dot(enrollment, difference)
    energy_difference = np.log(max(first_rms, 1e-6) / max(second_rms, 1e-6))
    return torch.cat((
        enrollment * difference,
        cosine_difference[None],
        difference.new_tensor([energy_difference]),
    )).cpu()


def collect_features(
    dataset: OpenYapWindows, separator, classifier, device: str, label: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    features = []
    labels = []
    rows = []
    separator_rate = int(separator.hparams.sample_rate)
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            mixture = sample["mixture"].numpy()
            target = sample["target"].numpy()
            other = sample["other"].numpy()
            if float(np.sqrt(np.mean(target**2))) < 1e-3:
                continue
            reduced = _resample(mixture, 16_000, separator_rate)
            separated = separator.separate_batch(
                torch.from_numpy(reduced)[None].to(device)
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
            enrollment = _speaker_embedding(
                classifier, trim_by_energy(sample["enrollment"].numpy()), device
            )
            embeddings = [
                _speaker_embedding(classifier, trim_by_energy(source), device)
                for source in sources
            ]
            if enrollment is None or any(embedding is None for embedding in embeddings):
                continue
            first_embedding, second_embedding = embeddings
            assert first_embedding is not None and second_embedding is not None
            first_rms, second_rms = (
                float(np.sqrt(np.mean(source**2))) for source in sources
            )
            features.append(feature_vector(
                enrollment, first_embedding, second_embedding, first_rms, second_rms
            ))
            scores = [score_extraction(source, target, other, 16_000) for source in sources]
            oracle = int(np.argmax([score.sisdr for score in scores]))
            labels.append(float(oracle == 0))
            short_scores = [
                short_overlap_turn_score(sample, source, target, other) for source in sources
            ]
            rows.append({
                "conversation_id": sample["conversation_id"],
                "target_side": sample["target_side"],
                "oracle": oracle,
                "overlap_s": scores[oracle].overlap_s,
                "source_sisdri_overlap": [
                    score.sisdri_overlap if np.isfinite(score.sisdri_overlap) else None
                    for score in scores
                ],
                "source_short_overlap_improvement_db": [
                    score[0] if score is not None else None for score in short_scores
                ],
            })
            if (index + 1) % 128 == 0:
                print(f"{label}: processed {index + 1}/{len(dataset)}", flush=True)
    if len(features) < 64:
        raise RuntimeError(f"Too few voiced {label} examples")
    return torch.stack(features), torch.tensor(labels), rows


def evaluate_selector(logits: torch.Tensor, rows: list[dict]) -> dict[str, float]:
    choices = (logits < 0).int().tolist()
    correct = [float(choice == row["oracle"]) for choice, row in zip(choices, rows, strict=True)]
    overlap_correct = [
        hit for hit, row in zip(correct, rows, strict=True) if row["overlap_s"] >= 0.1
    ]
    improvements = [
        row["source_sisdri_overlap"][choice]
        for choice, row in zip(choices, rows, strict=True)
    ]
    shorts = [
        row["source_short_overlap_improvement_db"][choice]
        for choice, row in zip(choices, rows, strict=True)
    ]
    metrics = {
        "assignment_accuracy": float(np.mean(correct)),
        "overlap_assignment_accuracy": float(np.mean(overlap_correct)),
        "sisdri_overlap": float(np.mean([value for value in improvements if value is not None])),
        "voiced_examples": float(len(correct)),
    }
    short_values = [value for value in shorts if value is not None]
    if short_values:
        metrics["short_overlap_improvement_db"] = float(np.mean(short_values))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--separator-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=2048)
    parser.add_argument("--val-examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="v2-learned-source-selector")
    parser.add_argument("--hf-repo")
    parser.add_argument("--no-tracker", action="store_true")
    args = parser.parse_args()
    if min(args.train_examples, args.val_examples) < 1 or args.enrollment_seconds <= 0:
        parser.error("Example counts and enrollment duration must be positive")
    if not args.no_tracker and (
        not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL")
    ):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.inference.separation import SepformerSeparation

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Training must use a clean commit")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.separator_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["source_repo"] != SEPARATOR_REPO:
        raise ValueError("Checkpoint separator repository does not match")
    checksum = hashlib.sha256()
    with args.separator_checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    separator = SepformerSeparation.from_hparams(
        source=SEPARATOR_REPO,
        savedir=str(args.out.parent / "pretrained" / SEPARATOR_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    separator.mods.load_state_dict(checkpoint["model"])
    classifier = EncoderClassifier.from_hparams(
        source=SPEAKER_REPO,
        savedir=str(args.out.parent / "pretrained" / SPEAKER_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    train_pairs, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    train_set = OpenYapWindows(
        train_pairs, args.train_examples, seed=4_242,
        enrollment_seconds=args.enrollment_seconds, overlap_sampling=0.5,
    )
    val_set = OpenYapWindows(
        val_pairs, args.val_examples, seed=10_042,
        enrollment_seconds=args.enrollment_seconds,
    )
    config = {
        "commit": commit,
        "separator_checkpoint_sha256": checksum.hexdigest(),
        "separator_step": checkpoint["step"],
        "separator_revision": checkpoint["source_revision"],
        "speaker_revision": HfApi().model_info(SPEAKER_REPO).sha,
        "train_examples": args.train_examples,
        "val_examples": args.val_examples,
        "enrollment_seconds": args.enrollment_seconds,
        "split_manifest": str(args.split_manifest),
    }
    tracker = None
    if not args.no_tracker:
        import airon

        tracker = airon.init(
            project="kezek", experiment="K-V2 recipe search on Open Yap",
            name=args.run_name, config=config,
        )
    try:
        train_x, train_y, train_rows = collect_features(
            train_set, separator, classifier, args.device, "train"
        )
        val_x, _, val_rows = collect_features(
            val_set, separator, classifier, args.device, "val"
        )
        scale = train_x.std(dim=0).clamp_min(1e-3)
        train_x = train_x / scale
        val_x = val_x / scale
        baseline = evaluate_selector(val_x[:, -2], val_rows)
        results = {"cosine": baseline}
        best = (baseline["sisdri_overlap"], "cosine", None)
        for penalty in (0.0001, 0.001, 0.01, 0.1):
            weights = torch.zeros(train_x.shape[1], requires_grad=True)
            optimizer = torch.optim.LBFGS([weights], lr=0.5, max_iter=200)

            def closure(optimizer=optimizer, weights=weights, penalty=penalty):
                optimizer.zero_grad()
                logits = train_x @ weights
                loss = F.binary_cross_entropy_with_logits(logits, train_y)
                loss = loss + penalty * weights.square().sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                metrics = evaluate_selector(val_x @ weights, val_rows)
            key = f"learned_{penalty}"
            results[key] = metrics
            if metrics["sisdri_overlap"] > best[0]:
                best = (metrics["sisdri_overlap"], key, weights.detach().clone())
        report = {
            "config": config,
            "metrics": results,
            "selected": best[1],
            "train_voiced_examples": len(train_rows),
            "val_rows": val_rows,
        }
        report_path = args.out / "report.json"
        report_path.write_text(json.dumps(report, indent=2))
        torch.save({
            "weights": best[2], "scale": scale, "method": best[1],
            "config": config,
        }, args.out / "selector.pt")
        print(json.dumps({"selected": best[1], "metrics": results}, sort_keys=True), flush=True)
        if tracker is not None:
            tracker.log({
                f"val/{method}/{metric}": value
                for method, measures in results.items()
                for metric, value in measures.items()
            }, step=0)
        if args.hf_repo:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required for upload")
            api = HfApi(token=token)
            if not api.repo_info(args.hf_repo).private:
                raise RuntimeError("Selector repository must be private")
            for path in (args.out / "selector.pt", report_path):
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"{args.run_name}/{path.name}",
                    repo_id=args.hf_repo,
                    commit_message=f"Save {args.run_name} {path.name}",
                )
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
