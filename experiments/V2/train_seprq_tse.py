"""Adapt a pretrained direct target extractor on natural Open Yap mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from evaluate_seprq import align_to_mixture, trim_by_energy
from train import git_state, select_splits, short_overlap_turn_score
from train_conditioned_separator import PairedTargetWindows
from train_sepformer import _si_sdr

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


def align_batch(estimate: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    """Apply the same mixture projection as at inference without losing gradients."""
    gain = (
        (mixture * estimate).sum(dim=-1)
        / estimate.square().sum(dim=-1).clamp_min(1e-8)
    ).clamp(-100, 100)
    return estimate * gain[:, None]


def target_loss(
    estimate: torch.Tensor, target: torch.Tensor, other: torch.Tensor,
    mixture: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if estimate.shape != target.shape or target.shape != other.shape:
        raise ValueError("Estimated and reference waveform shapes do not match")
    aligned = align_batch(estimate, mixture)
    normalizer = mixture.square().mean(dim=-1).clamp_min(1e-6)
    mse = ((aligned - target).square().mean(dim=-1) / normalizer).mean()
    sisdr = _si_sdr(estimate, target).mean()
    frame = 320
    frames = target.shape[-1] // frame
    target_energy = (
        target[:, : frames * frame].reshape(len(target), frames, frame).square().mean(-1)
    )
    other_energy = other[:, : frames * frame].reshape(len(other), frames, frame).square().mean(-1)
    overlap = (target_energy >= 1e-6) & (other_energy >= 1e-6)
    overlap_samples = overlap.repeat_interleave(frame, dim=-1)
    overlap_error = (aligned[:, : frames * frame] - target[:, : frames * frame]).square()
    overlap_loss = (
        (overlap_error * overlap_samples).sum(dim=-1)
        / overlap_samples.sum(dim=-1).clamp_min(1)
        / normalizer
    ).mean()
    loss = mse + 0.5 * overlap_loss - 0.05 * sisdr
    return loss, {
        "target_mse": float(mse.detach()),
        "overlap_mse": float(overlap_loss.detach()),
        "target_sisdr": float(sisdr.detach()),
        "overlap_frames": float(overlap.float().sum().detach()),
    }


def evaluate(pipe, dataset) -> tuple[dict, list[dict]]:
    pipe.head.eval()
    scores = []
    shorts = []
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            mixture = sample["mixture"]
            enrollment = torch.from_numpy(
                trim_by_energy(sample["enrollment"].numpy()).copy()
            )
            output = pipe(mixture, enrollment=enrollment).detach().float().cpu().numpy()
            target = sample["target"].numpy()
            other = sample["other"].numpy()
            output = np.pad(output[: len(target)], (0, max(0, len(target) - len(output))))
            estimate = align_to_mixture(output, mixture.numpy())
            score = score_extraction(estimate, target, other, 16_000)
            scores.append(score.as_dict())
            short = short_overlap_turn_score(sample, estimate, target, other)
            if short is not None:
                shorts.append(short[0])
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
    if shorts:
        metrics["short_overlap_turn_improvement_db"] = float(np.mean(shorts))
        metrics["short_overlap_turn_examples"] = float(len(shorts))
    pipe.head.train()
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name", default="v2-direct-tse-adaptation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.val_examples) < 1:
        parser.error("Steps, batch size and validation count must be positive")
    if args.lr <= 0 or args.workers < 0 or args.enrollment_seconds <= 0:
        parser.error("Learning rate, workers or enrollment duration are invalid")
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from seprq import SepRQPipeline

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Training must use a clean commit")
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = SepRQPipeline("target-speaker-extraction", upstream="WavLM_BASE", device=args.device)
    pipe.encoder.eval()
    pipe.head.train()
    optimizer = torch.optim.AdamW(pipe.head.parameters(), lr=args.lr, weight_decay=1e-4)
    resume_step = 0
    resume_checksum = None
    if args.resume:
        checksum = hashlib.sha256()
        with args.resume.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        resume_checksum = checksum.hexdigest()
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["config"]["task"] != "target-speaker-extraction":
            raise ValueError("Resume checkpoint is not a target extraction model")
        pipe.head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        resume_step = int(state["step"])
        if resume_step >= args.steps:
            raise ValueError("Resume step must be below requested total steps")
    train_pairs, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    natural = OpenYapWindows(
        train_pairs, (args.steps * args.batch_size + 1) // 2, seed=37_821,
        enrollment_seconds=args.enrollment_seconds, overlap_sampling=0.5,
    )
    train_set = PairedTargetWindows(natural)
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
        "package_version": version("seprq"),
        "model_revision": HfApi().model_info("SepRQ/SepRQ").sha,
        "task": "target-speaker-extraction",
        "upstream": "WavLM_BASE",
        "split_manifest": str(args.split_manifest),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "resume_step": resume_step,
        "resume_sha256": resume_checksum,
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
        baseline, rows = evaluate(pipe, val_set)
        print(f"eval step {resume_step}: {json.dumps(baseline, sort_keys=True)}", flush=True)
        tracker.log({f"val/{key}": value for key, value in baseline.items()}, step=resume_step)
        (args.out / f"validation_{resume_step:06d}.json").write_text(
            json.dumps({"metrics": baseline, "rows": rows}, indent=2)
        )
        for step, sample in enumerate(loader, 1):
            if step <= resume_step:
                continue
            mixture = sample["mixture"].to(args.device)
            target = sample["target"].to(args.device)
            other = sample["other"].to(args.device)
            enrollment = [
                torch.from_numpy(trim_by_energy(wave.numpy()).copy())
                for wave in sample["enrollment"]
            ]
            estimate = pipe(mixture, enrollment=enrollment)
            loss, measures = target_loss(estimate, target, other, mixture)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite direct TSE loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.head.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == resume_step + 1 or step % 20 == 0:
                payload = {"train/loss": float(loss.detach())}
                payload.update({f"train/{key}": value for key, value in measures.items()})
                tracker.log(payload, step=step)
                print(f"step {step}: {payload}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, rows = evaluate(pipe, val_set)
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                torch.save({
                    "head": pipe.head.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": config,
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
