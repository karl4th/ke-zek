"""Train and evaluate the direct extractor on aligned Open Yap conversations.

The test split is recorded but never used for model selection. A paid run should
use a clean git commit, Airon tracking, and persistent checkpoint storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations, split_conversations
from kezek.open_yap_training import SAMPLE_RATE, OpenYapWindows
from kezek.tse_v2 import TargetSpeakerExtractor
from kezek.tse_v2_loss import extraction_loss


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stop-at", type=int, default=0, help="Profile a prefix, then resume")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mixture-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=6.0)
    parser.add_argument(
        "--overlap-sampling",
        type=float,
        default=0.0,
        help="Chance of centering a training crop on word overlap",
    )
    parser.add_argument("--val-examples", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-train-conversations", type=int, default=0)
    parser.add_argument("--max-val-conversations", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="kezek-v2-open-yap")
    parser.add_argument("--hf-repo", help="Existing private model repository for checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--no-tracker", action="store_true", help="Local smoke checks only")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.val_examples, args.eval_every) < 1:
        parser.error("Steps, batch size, validation examples, and eval interval must be positive")
    if args.workers < 0:
        parser.error("Workers must be nonnegative")
    if args.stop_at < 0 or args.stop_at > args.steps:
        parser.error("Stop step must be between zero and total steps")
    if not 0 <= args.overlap_sampling <= 1:
        parser.error("Overlap sampling must be between zero and one")
    return args


def git_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    return commit, dirty


def limited(pairs, maximum: int, seed: int):
    if maximum <= 0 or len(pairs) <= maximum:
        return pairs
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(pairs), maximum, replace=False))
    return [pairs[int(i)] for i in indices]


def init_tracker(args: argparse.Namespace, manifest: dict):
    if args.no_tracker:
        return None
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        raise RuntimeError("AIRON_API_KEY and AIRON_URL are required unless --no-tracker is set")
    import airon

    split_bytes = json.dumps(manifest["splits"], sort_keys=True).encode()
    tracker_config = {
        **manifest["config"],
        "commit": manifest["commit"],
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "split_counts": {name: len(items) for name, items in manifest["splits"].items()},
        "stop_at": args.stop_at,
    }

    return airon.init(
        project="kezek",
        name=args.run_name,
        experiment="K-V2",
        config=tracker_config,
        tags=["v2", "open-yap", "tse"],
        total_steps=args.steps,
    )


def upload_checkpoint(args: argparse.Namespace, paths: list[Path]) -> None:
    if not args.hf_repo:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for checkpoint uploads")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if not api.repo_info(args.hf_repo).private:
        raise RuntimeError("Checkpoint repository must be private")
    prefix = args.run_name.strip("/")
    for path in paths:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"{prefix}/{path.name}",
            repo_id=args.hf_repo,
            commit_message=f"Save {prefix} {path.name}",
        )
    remote = {
        item.rfilename: item.size
        for item in api.repo_info(args.hf_repo, files_metadata=True).siblings
    }
    if any(remote.get(f"{prefix}/{path.name}") != path.stat().st_size for path in paths):
        raise RuntimeError("Uploaded checkpoint files have unexpected sizes")


def evaluate(model, dataset, device, demo_dir: Path, demo_count: int = 4) -> dict[str, float]:
    model.eval()
    scores: list[dict[str, float]] = []
    demo_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            mixture = sample["mixture"][None].to(device)
            enrollment = sample["enrollment"][None].to(device)
            estimate = model(mixture, enrollment).waveform[0].cpu().numpy()
            target = sample["target"].numpy()
            other = sample["other"].numpy()
            scores.append(score_extraction(estimate, target, other, SAMPLE_RATE).as_dict())
            if index < demo_count:
                for name, audio in (
                    ("mixture", sample["mixture"].numpy()),
                    ("target", target),
                    ("estimate", estimate),
                ):
                    sf.write(
                        demo_dir / f"{index:03d}_{name}.wav", audio, SAMPLE_RATE, subtype="FLOAT"
                    )
    model.train()
    return {
        key: float(np.mean(values))
        for key in scores[0]
        if (values := [score[key] for score in scores if np.isfinite(score[key])])
    }


def save_checkpoint(path: Path, step: int, model, optimizer, scheduler, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "manifest": manifest,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse()
    if not args.no_tracker and not args.hf_repo:
        raise RuntimeError("A tracked run requires --hf-repo for persistent checkpoints")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    commit, dirty = git_state()
    if args.require_clean and dirty:
        raise RuntimeError("A paid run must use a clean, committed repository")

    pairs = scan_conversations(args.data_root)
    train_pairs, val_pairs, test_pairs = split_conversations(pairs, seed=args.seed)
    train_pairs = limited(train_pairs, args.max_train_conversations, args.seed)
    val_pairs = limited(val_pairs, args.max_val_conversations, args.seed + 1)
    manifest = {
        "commit": commit,
        "dirty": dirty,
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in {"resume", "stop_at", "no_tracker", "require_clean"}
        },
        "splits": {
            name: [pair.conversation_id for pair in group]
            for name, group in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))
        },
    }
    manifest_path = args.out / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        if previous["splits"] != manifest["splits"] or previous["config"] != manifest["config"]:
            raise RuntimeError("Resume settings or data split changed")
        manifest = previous
    else:
        if manifest_path.exists():
            raise RuntimeError("Output already has a manifest; use --resume or a new --out")
        manifest_path.write_text(json.dumps(manifest, indent=2))

    train_set = OpenYapWindows(
        train_pairs,
        args.steps * args.batch_size,
        args.seed,
        args.mixture_seconds,
        args.enrollment_seconds,
        overlap_sampling=args.overlap_sampling,
    )
    val_set = OpenYapWindows(
        val_pairs,
        args.val_examples,
        args.seed + 10_000,
        args.mixture_seconds,
        args.enrollment_seconds,
    )
    model = TargetSpeakerExtractor(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        bands=args.bands,
        width=args.width,
        blocks=args.blocks,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup = min(100, max(1, args.steps // 20))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, args.steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    checkpoint_path = args.out / "last.pt"
    start_step = 0
    if args.resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["manifest"]["splits"] != manifest["splits"]:
            raise RuntimeError("Checkpoint data split changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]

    end_step = args.stop_at or args.steps
    if start_step >= end_step:
        raise RuntimeError(f"Checkpoint is already at step {start_step}")

    remaining = torch.utils.data.Subset(
        train_set, range(start_step * args.batch_size, len(train_set))
    )
    loader = torch.utils.data.DataLoader(
        remaining,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=2 if args.workers else None,
    )
    tracker = init_tracker(args, manifest)
    try:
        if start_step == 0:
            baseline = evaluate(model, val_set, args.device, args.out / "audio" / "step_0")
            print(f"eval step 0: {json.dumps(baseline, sort_keys=True)}", flush=True)
            if tracker is not None:
                tracker.log({f"val/{key}": value for key, value in baseline.items()}, step=0)
        for step, sample in enumerate(loader, start_step + 1):
            mixture = sample["mixture"].to(args.device, non_blocking=True)
            enrollment = sample["enrollment"].to(args.device, non_blocking=True)
            target = sample["target"].to(args.device, non_blocking=True)
            other = sample["other"].to(args.device, non_blocking=True)
            output = model(mixture, enrollment)
            loss, parts = extraction_loss(output, target, other)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if tracker is not None and (step % 10 == 0 or step == 1):
                tracker.log(
                    {
                        "train/loss": float(loss.detach()),
                        **{f"train/{key}": value for key, value in parts.items()},
                    },
                    step=step,
                )
            if step == 1 or step % 10 == 0:
                print(f"step {step}: loss={float(loss.detach()):.3f} {parts}", flush=True)
            if step % args.eval_every == 0 or step == end_step:
                metrics = evaluate(model, val_set, args.device, args.out / "audio" / f"step_{step}")
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                if tracker is not None:
                    tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                save_checkpoint(checkpoint_path, step, model, optimizer, scheduler, manifest)
                model_path = args.out / "model.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": {
                            "n_fft": args.n_fft,
                            "hop_length": args.hop_length,
                            "bands": args.bands,
                            "width": args.width,
                            "blocks": args.blocks,
                        },
                    },
                    model_path,
                )
                upload_checkpoint(args, [checkpoint_path, model_path, manifest_path])
            if step == end_step:
                break
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
