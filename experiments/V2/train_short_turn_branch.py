"""Train a temporal mask correction with the established extractor held fixed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from evaluate_sepformer import trim_by_energy
from evaluate_seprq import load_head_checkpoint
from short_turn_branch import attach_branch
from train import git_state, select_splits, upload_checkpoint
from train_conditioned_separator import PairedTargetWindows
from train_seprq_tse import evaluate, target_loss

from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-head-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--run-name", default="v2-temporal-mask-correction-2k")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.val_examples, args.width) < 1:
        parser.error("Steps, batch size, evaluation interval, count, and width must be positive")
    if args.lr <= 0 or args.workers < 0 or args.enrollment_seconds <= 0:
        parser.error("Learning rate, workers, and enrollment duration are invalid")
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from seprq import SepRQPipeline

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("Training must use a clean commit")
    if not HfApi().repo_info(args.hf_repo).private:
        raise RuntimeError("Checkpoint repository must be private")
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = SepRQPipeline("target-speaker-extraction", upstream="WavLM_BASE", device=args.device)
    base_hash, base_step = load_head_checkpoint(
        pipe, args.base_head_checkpoint, "target-speaker-extraction", "WavLM_BASE"
    )
    pipe.requires_grad_(False).eval()
    branch = attach_branch(pipe.head, args.width)
    branch.train()
    optimizer = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)

    train_pairs, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    natural = OpenYapWindows(
        train_pairs, (args.steps * args.batch_size + 1) // 2, seed=43_821,
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
        "architecture": "gated-multiscale-temporal-mask-correction",
        "base_head_sha256": base_hash,
        "base_head_step": base_step,
        "frozen_base": True,
        "trainable_parameters": sum(p.numel() for p in branch.parameters() if p.requires_grad),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "width": args.width,
        "val_examples": args.val_examples,
        "train_conversations": len(train_pairs),
        "val_conversations": len(val_pairs),
        "enrollment_seconds": args.enrollment_seconds,
        "checkpoint_storage": args.hf_repo,
    }
    (args.out / "manifest.json").write_text(json.dumps(config, indent=2))
    import airon

    tracker = airon.init(
        project="kezek", experiment="K-V2 recipe search on Open Yap",
        name=args.run_name, config=config, total_steps=args.steps,
    )
    try:
        baseline, rows = evaluate(pipe, val_set)
        pipe.head.eval()
        branch.train()
        print(f"eval step 0: {json.dumps(baseline, sort_keys=True)}", flush=True)
        tracker.log({f"val/{key}": value for key, value in baseline.items()}, step=0)
        (args.out / "validation_000000.json").write_text(
            json.dumps({"metrics": baseline, "rows": rows}, indent=2)
        )
        for step, sample in enumerate(loader, 1):
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
                raise RuntimeError(f"Nonfinite temporal correction loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 20 == 0:
                payload = {"train/loss": float(loss.detach())}
                payload.update({f"train/{key}": value for key, value in measures.items()})
                tracker.log(payload, step=step)
                print(f"step {step}: {payload}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, rows = evaluate(pipe, val_set)
                pipe.head.eval()
                branch.train()
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                checkpoint = args.out / f"step_{step:06d}.pt"
                torch.save({
                    "branch": branch.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": config,
                }, checkpoint)
                (args.out / f"validation_{step:06d}.json").write_text(
                    json.dumps({"metrics": metrics, "rows": rows}, indent=2)
                )
                upload_checkpoint(args, [checkpoint, args.out / "manifest.json"])
            if step == args.steps:
                break
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
