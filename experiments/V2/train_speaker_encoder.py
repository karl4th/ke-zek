"""Adapt a pretrained voice encoder with clean Open Yap speaker pairs."""

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
from evaluate_sepformer import SPEAKER_REPO, _voice_score, trim_by_energy
from train import git_state, select_splits, upload_checkpoint
from train_sepformer import SOURCE_REPO, align_sources_to_mixture

from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


def normalize_waveform(audio: torch.Tensor) -> torch.Tensor:
    rms = audio.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
    return audio * (0.1 / rms)


def speaker_pair_loss(
    model, enrollment: torch.Tensor, target: torch.Tensor, other: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    reference = F.normalize(model.encode_batch(normalize_waveform(enrollment)).squeeze(1), dim=-1)
    same = F.normalize(model.encode_batch(normalize_waveform(target)).squeeze(1), dim=-1)
    different = F.normalize(model.encode_batch(normalize_waveform(other)).squeeze(1), dim=-1)
    positive = (reference * same).sum(dim=-1)
    negative = (reference * different).sum(dim=-1)
    loss = F.softplus((negative - positive) / 0.1).mean()
    return loss, {
        "positive_cosine": float(positive.mean().detach()),
        "negative_cosine": float(negative.mean().detach()),
        "pair_accuracy": float((positive > negative).float().mean().detach()),
    }


def separated_pairs(
    separator, mixture: torch.Tensor, target: torch.Tensor, other: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use clean references to identify both frozen separator outputs for training."""
    with torch.no_grad():
        reduced_mix = AF.resample(mixture, 16_000, 8_000)
        reduced_target = AF.resample(target, 16_000, 8_000)
        reduced_other = AF.resample(other, 16_000, 8_000)
        estimates, _ = align_sources_to_mixture(
            separator.separate_batch(reduced_mix), reduced_mix
        )
        first, second = estimates[..., 0], estimates[..., 1]
        direct = (first - reduced_target).square().mean(dim=-1)
        direct += (second - reduced_other).square().mean(dim=-1)
        swapped = (second - reduced_target).square().mean(dim=-1)
        swapped += (first - reduced_other).square().mean(dim=-1)
        use_second = swapped < direct
        positive = torch.where(use_second[:, None], second, first)
        negative = torch.where(use_second[:, None], first, second)
        positive = AF.resample(positive, 8_000, 16_000)
        negative = AF.resample(negative, 8_000, 16_000)
    return positive, negative


def evaluate(model, dataset: OpenYapWindows, device: str) -> dict[str, float]:
    model.mods.eval()
    accuracy = []
    margins = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            target = sample["target"].numpy()
            if float(np.sqrt(np.mean(target**2))) < 1e-3:
                continue
            enrollment = trim_by_energy(sample["enrollment"].numpy())
            other = trim_by_energy(sample["other"].numpy())
            target = trim_by_energy(target)
            same = _voice_score(model, enrollment, target, device)
            different = _voice_score(model, enrollment, other, device)
            accuracy.append(float(same > different))
            margins.append(same - different)
    if not accuracy:
        raise RuntimeError("Validation has no voiced target examples")
    return {
        "clean_speaker_assignment_accuracy": float(np.mean(accuracy)),
        "clean_speaker_margin": float(np.mean(margins)),
        "clean_speaker_examples": float(len(accuracy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--separator-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="v2-speaker-encoder-adaptation")
    parser.add_argument("--hf-repo", required=True)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.val_examples) < 1:
        parser.error("Steps, batch size, and validation settings must be positive")
    if args.enrollment_seconds <= 0 or args.lr <= 0 or args.workers < 0:
        parser.error("Enrollment, learning rate, and worker settings are invalid")
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from speechbrain.inference.classifiers import EncoderClassifier

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Training must use a clean commit")
    args.out.mkdir(parents=True, exist_ok=True)
    model_revision = HfApi().model_info(SPEAKER_REPO).sha
    model = EncoderClassifier.from_hparams(
        source=SPEAKER_REPO,
        savedir=str(args.out.parent / "pretrained" / SPEAKER_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    separator = None
    separator_checksum = None
    separator_step = None
    if args.separator_checkpoint:
        checksum = hashlib.sha256()
        with args.separator_checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        separator_checksum = checksum.hexdigest()
        separator_state = torch.load(
            args.separator_checkpoint, map_location="cpu", weights_only=True
        )
        if separator_state["source_repo"] != SOURCE_REPO:
            raise ValueError("Separator checkpoint repository does not match")
        separator_step = separator_state["step"]
        from speechbrain.inference.separation import SepformerSeparation

        separator = SepformerSeparation.from_hparams(
            source=SOURCE_REPO,
            savedir=str(args.out.parent / "pretrained" / SOURCE_REPO.replace("/", "_")),
            run_opts={"device": args.device},
        )
        separator.mods.load_state_dict(separator_state["model"])
        separator.mods.eval()
    model.mods.eval()
    for parameter in model.mods.parameters():
        parameter.requires_grad_(False)
    encoder = model.mods.embedding_model
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    train_pairs, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    train_set = OpenYapWindows(
        train_pairs, args.steps * args.batch_size, seed=7_521,
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
        "speaker_repo": SPEAKER_REPO,
        "speaker_revision": model_revision,
        "separator_checkpoint_sha256": separator_checksum,
        "separator_step": separator_step,
        "split_manifest": str(args.split_manifest),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_every": args.eval_every,
        "val_examples": args.val_examples,
        "enrollment_seconds": args.enrollment_seconds,
        "train_conversations": len(train_pairs),
        "val_conversations": len(val_pairs),
    }
    (args.out / "manifest.json").write_text(json.dumps(config, indent=2))
    import airon

    tracker = airon.init(
        project="kezek", experiment="K-V2 recipe search on Open Yap",
        name=args.run_name, config=config, total_steps=args.steps,
    )
    best_accuracy = -1.0
    try:
        baseline = evaluate(model, val_set, args.device)
        print(f"eval step 0: {json.dumps(baseline, sort_keys=True)}", flush=True)
        tracker.log({f"val/{key}": value for key, value in baseline.items()}, step=0)
        best_accuracy = baseline["clean_speaker_assignment_accuracy"]
        torch.save({
            "model": encoder.state_dict(), "step": 0,
            "speaker_repo": SPEAKER_REPO, "speaker_revision": model_revision,
        }, args.out / "best_model.pt")
        upload_checkpoint(args, [args.out / "best_model.pt", args.out / "manifest.json"])
        for step, sample in enumerate(loader, 1):
            enrollment = sample["enrollment"].to(args.device, non_blocking=True)
            target = sample["target"].to(args.device, non_blocking=True)
            other = sample["other"].to(args.device, non_blocking=True)
            usable = (
                target.square().mean(dim=-1) >= 1e-6
            ) & (
                other.square().mean(dim=-1) >= 1e-6
            )
            if int(usable.sum()) >= 2:
                positive, negative = target[usable], other[usable]
                if separator is not None:
                    positive, negative = separated_pairs(
                        separator, sample["mixture"].to(args.device)[usable],
                        positive, negative,
                    )
                encoder.train()
                loss, metrics = speaker_pair_loss(
                    model, enrollment[usable], positive, negative
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite speaker loss at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 20 == 0:
                    payload = {
                        "train/loss": float(loss.detach()),
                        "train/usable": float(usable.sum()),
                    }
                    payload.update({f"train/{key}": value for key, value in metrics.items()})
                    tracker.log(payload, step=step)
                    print(f"step {step}: {payload}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, val_set, args.device)
                tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                payload = {
                    "model": encoder.state_dict(), "step": step,
                    "speaker_repo": SPEAKER_REPO, "speaker_revision": model_revision,
                }
                torch.save(payload, args.out / "model.pt")
                paths = [args.out / "model.pt", args.out / "manifest.json"]
                if metrics["clean_speaker_assignment_accuracy"] > best_accuracy:
                    best_accuracy = metrics["clean_speaker_assignment_accuracy"]
                    torch.save(payload, args.out / "best_model.pt")
                    paths.append(args.out / "best_model.pt")
                torch.save({
                    **payload, "optimizer": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                }, args.out / "last.pt")
                paths.append(args.out / "last.pt")
                upload_checkpoint(args, paths)
            if step == args.steps:
                break
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
