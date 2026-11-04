"""Train a voice-conditioned frame router over frozen two-source separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from evaluate_sepformer import (
    SEPARATOR_REPO, SPEAKER_REPO, _speaker_embedding, trim_by_energy,
)
from train import git_state, select_splits, short_overlap_turn_score
from train_sepformer import _si_sdr, align_sources_to_mixture

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows
from kezek.target_router import TargetRouter, oracle_route_labels


def speaker_features(classifier, enrollment, sources, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed clean enrollment and separated sources without updating the encoder."""
    batch_enrollment = []
    batch_sources = []
    with torch.inference_mode():
        upsampled = AF.resample(sources.transpose(1, 2), 8_000, 16_000)
        for index in range(len(sources)):
            reference = trim_by_energy(enrollment[index].detach().cpu().numpy())
            voice = _speaker_embedding(classifier, reference, device)
            if voice is None:
                voice = torch.zeros(192, device=device)
            source_voices = []
            for side in range(2):
                audio = trim_by_energy(upsampled[index, side].detach().cpu().numpy())
                embedding = _speaker_embedding(classifier, audio, device)
                source_voices.append(
                    embedding if embedding is not None else torch.zeros_like(voice)
                )
            batch_enrollment.append(voice)
            batch_sources.append(torch.stack(source_voices))
    return torch.stack(batch_enrollment), torch.stack(batch_sources)


def prepare_batch(separator, classifier, sample: dict, device: str):
    mixture = sample["mixture"].to(device)
    target = sample["target"].to(device)
    reduced_mixture = AF.resample(mixture, 16_000, 8_000)
    reduced_target = AF.resample(target, 16_000, 8_000)
    with torch.inference_mode():
        sources, _ = align_sources_to_mixture(
            separator.separate_batch(reduced_mixture), reduced_mixture
        )
        voice, source_voices = speaker_features(
            classifier, sample["enrollment"], sources, device
        )
    return (
        sources.detach().clone(), reduced_target, reduced_mixture,
        voice.detach().clone(), source_voices.detach().clone(),
    )


def routing_loss(
    estimate: torch.Tensor, logits: torch.Tensor, sources: torch.Tensor,
    target: torch.Tensor, mixture: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = oracle_route_labels(sources, target, logits.shape[-1])
    class_counts = torch.bincount(labels.flatten(), minlength=3).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1)).sqrt()
    class_weights = class_weights / class_weights.mean()
    route = F.cross_entropy(logits, labels, weight=class_weights)
    mse = (
        (estimate - target).square().mean(dim=-1)
        / mixture.square().mean(dim=-1).clamp_min(1e-6)
    ).mean()
    sisdr = _si_sdr(estimate, target).mean()
    loss = 0.5 * route + mse - 0.05 * sisdr
    return loss, {
        "route_ce": float(route.detach()),
        "target_mse": float(mse.detach()),
        "target_sisdr": float(sisdr.detach()),
        "oracle_route_accuracy": float((logits.argmax(1) == labels).float().mean().detach()),
    }


def evaluate(router, separator, classifier, dataset, device: str) -> tuple[dict, list[dict]]:
    router.eval()
    scores = []
    oracle_scores = []
    short_scores = []
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            batched = {
                key: value[None] if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }
            sources, _, _, voice, source_voices = prepare_batch(
                separator, classifier, batched, device
            )
            estimate, _ = router(sources, voice, source_voices)
            estimate = AF.resample(estimate, 8_000, 16_000)[0].cpu().numpy()
            target = sample["target"].numpy()
            other = sample["other"].numpy()
            estimate = np.pad(estimate[: len(target)], (0, max(0, len(target) - len(estimate))))
            score = score_extraction(estimate, target, other, 16_000)
            scores.append(score.as_dict())
            reference_sources = AF.resample(
                sources.transpose(1, 2), 8_000, 16_000
            )[0].cpu().numpy()
            source_scores = [
                score_extraction(source[: len(target)], target, other, 16_000)
                for source in reference_sources
            ]
            oracle_scores.append(max(
                source_scores, key=lambda item: item.sisdr
            ).sisdri_overlap)
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
        if (values := [row[key] for row in scores if np.isfinite(row[key])])
    }
    metrics["oracle_sisdri_overlap"] = float(np.mean([
        value for value in oracle_scores if np.isfinite(value)
    ]))
    if short_scores:
        metrics["short_overlap_turn_improvement_db"] = float(np.mean(short_scores))
        metrics["short_overlap_turn_examples"] = float(len(short_scores))
    router.train()
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--separator-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument("--run-name", default="v2-target-router")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.val_examples) < 1:
        parser.error("Steps, batch size and validation count must be positive")
    if args.lr <= 0 or args.workers < 0 or args.enrollment_seconds <= 0:
        parser.error("Learning rate, workers or enrollment duration are invalid")
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.inference.separation import SepformerSeparation

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Training must use a clean commit")
    args.out.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    with args.separator_checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    state = torch.load(args.separator_checkpoint, map_location="cpu", weights_only=True)
    if state["source_repo"] != SEPARATOR_REPO:
        raise ValueError("Separator checkpoint repository does not match")
    separator = SepformerSeparation.from_hparams(
        source=SEPARATOR_REPO,
        savedir=str(args.out.parent / "pretrained" / SEPARATOR_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    separator.mods.load_state_dict(state["model"])
    separator.mods.eval()
    classifier = EncoderClassifier.from_hparams(
        source=SPEAKER_REPO,
        savedir=str(args.out.parent / "pretrained" / SPEAKER_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    classifier.mods.eval()
    router = TargetRouter().to(args.device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    train_pairs, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    train_set = OpenYapWindows(
        train_pairs, args.steps * args.batch_size, seed=18_721,
        enrollment_seconds=args.enrollment_seconds, overlap_sampling=0.5,
    )
    val_set = OpenYapWindows(
        val_pairs, args.val_examples, seed=10_042,
        enrollment_seconds=args.enrollment_seconds,
    )
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=2 if args.workers else None,
    )
    config = {
        "commit": commit,
        "separator_step": state["step"],
        "separator_sha256": checksum.hexdigest(),
        "separator_revision": state["source_revision"],
        "speaker_repo": SPEAKER_REPO,
        "speaker_revision": HfApi().model_info(SPEAKER_REPO).sha,
        "split_manifest": str(args.split_manifest),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_examples": args.val_examples,
        "enrollment_seconds": args.enrollment_seconds,
        "train_conversations": len(train_pairs),
        "val_conversations": len(val_pairs),
        "checkpoint_storage": "local",
    }
    (args.out / "manifest.json").write_text(json.dumps(config, indent=2))
    import airon

    tracker = airon.init(
        project="kezek", experiment="K-V2 recipe search on Open Yap",
        name=args.run_name, config=config, total_steps=args.steps,
    )
    try:
        baseline, rows = evaluate(router, separator, classifier, val_set, args.device)
        print(f"eval step 0: {json.dumps(baseline, sort_keys=True)}", flush=True)
        tracker.log({f"val/{key}": value for key, value in baseline.items()}, step=0)
        (args.out / "validation_000000.json").write_text(
            json.dumps({"metrics": baseline, "rows": rows}, indent=2)
        )
        for step, sample in enumerate(loader, 1):
            sources, target, mixture, voice, source_voices = prepare_batch(
                separator, classifier, sample, args.device
            )
            estimate, logits = router(sources, voice, source_voices)
            loss, measures = routing_loss(estimate, logits, sources, target, mixture)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite router loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 20 == 0:
                payload = {"train/loss": float(loss.detach())}
                payload.update({f"train/{key}": value for key, value in measures.items()})
                tracker.log(payload, step=step)
                print(f"step {step}: {payload}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, rows = evaluate(
                    router, separator, classifier, val_set, args.device
                )
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                torch.save({
                    "model": router.state_dict(), "step": step,
                    "config": config,
                }, args.out / f"step_{step:06d}.pt")
                (args.out / f"validation_{step:06d}.json").write_text(
                    json.dumps({"metrics": metrics, "rows": rows}, indent=2)
                )
            if step == args.steps:
                break
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
