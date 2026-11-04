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
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from kezek.metrics import score_extraction
from kezek.open_yap import check_speaker_disjoint, scan_conversations, split_conversations
from kezek.open_yap_training import SAMPLE_RATE, CrossPairWindows, OpenYapWindows
from kezek.tse_v2 import TargetSpeakerExtractor
from kezek.tse_v2_loss import (
    extraction_loss,
    mask_supervision_loss,
    source_discriminative_loss,
)
from kezek.tse_v2_unet import SpectralUNetExtractor


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Reuse the exact train/val/test conversation IDs from a previous run",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stop-at", type=int, default=0, help="Profile a prefix, then resume")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mixture-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=6.0)
    parser.add_argument(
        "--enrollment-selection",
        choices=("ratio", "power"),
        default="ratio",
        help="Choose the loudest clean target sample, or the old target/other power ratio",
    )
    parser.add_argument(
        "--overlap-sampling",
        type=float,
        default=0.0,
        help="Chance of centering a training crop on word overlap",
    )
    parser.add_argument(
        "--paired-directions",
        action="store_true",
        help="Train both speaker directions on each mixture when both voice samples are usable",
    )
    parser.add_argument("--val-examples", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-train-conversations", type=int, default=0)
    parser.add_argument("--max-val-conversations", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--architecture", choices=("band", "spectral_unet"), default="band")
    parser.add_argument("--speaker-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--source-discriminative-weight", type=float, default=0.0)
    parser.add_argument("--cross-pair-probability", type=float, default=0.0)
    parser.add_argument("--mask-supervision-weight", type=float, default=0.0)
    parser.add_argument("--conditioning", choices=("mean", "band_concat"), default="mean")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="kezek-v2-open-yap")
    parser.add_argument("--hf-repo", help="Existing private model repository for checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-model", type=Path, help="Initialize a new run from model.pt")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--no-tracker", action="store_true", help="Local smoke checks only")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.val_examples, args.eval_every) < 1:
        parser.error("Steps, batch size, validation examples, and eval interval must be positive")
    if args.workers < 0:
        parser.error("Workers must be nonnegative")
    if args.stop_at < 0 or args.stop_at > args.steps:
        parser.error("Stop step must be between zero and total steps")
    if args.resume and args.init_model:
        parser.error("--resume and --init-model cannot be combined")
    if not 0 <= args.overlap_sampling <= 1:
        parser.error("Overlap sampling must be between zero and one")
    if not 0 <= args.cross_pair_probability <= 1:
        parser.error("Cross-pair probability must be between zero and one")
    if args.speaker_contrastive_weight < 0:
        parser.error("Speaker contrastive weight must be nonnegative")
    if args.source_discriminative_weight < 0:
        parser.error("Source discriminative weight must be nonnegative")
    if args.mask_supervision_weight < 0:
        parser.error("Mask supervision weight must be nonnegative")
    if args.speaker_contrastive_weight and args.architecture != "spectral_unet":
        parser.error("Speaker contrastive loss requires spectral_unet")
    return args


def git_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    return commit, dirty


def load_initial_model(model, path: Path, architecture: str, model_config: dict) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("architecture", "band") != architecture or state["model_config"] != model_config:
        raise ValueError("Initial model architecture or settings do not match")
    model.load_state_dict(state["model"])


def limited(pairs, maximum: int, seed: int):
    if maximum <= 0 or len(pairs) <= maximum:
        return pairs
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(pairs), maximum, replace=False))
    return [pairs[int(i)] for i in indices]


def select_splits(pairs, split_manifest: Path | None, seed: int):
    if split_manifest is None:
        return split_conversations(pairs, seed=seed)
    previous = json.loads(split_manifest.read_text())["splits"]
    by_id = {pair.conversation_id: pair for pair in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("Duplicate conversation IDs in data root")
    selected = []
    used = set()
    for name in ("train", "val", "test"):
        ids = previous[name]
        if not ids or len(ids) != len(set(ids)) or used.intersection(ids):
            raise ValueError(f"Invalid {name} IDs in split manifest")
        missing = set(ids) - by_id.keys()
        if missing:
            raise ValueError(f"Missing {name} conversations: {sorted(missing)[:3]}")
        selected.append([by_id[conversation_id] for conversation_id in ids])
        used.update(ids)
    check_speaker_disjoint(*selected)
    return tuple(selected)


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
        experiment="K-V2 recipe search on Open Yap",
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


@lru_cache(maxsize=512)
def transcript_words(path: Path) -> tuple[tuple[float, float], ...]:
    words = json.loads(path.read_text())["words"]
    return tuple(sorted((float(word["start"]), float(word["end"])) for word in words))


def short_overlap_turn_score(
    sample: dict, estimate: np.ndarray, target: np.ndarray, other: np.ndarray
) -> tuple[float, int] | None:
    """Score isolated short target turns during transcript overlap against mixture error."""
    if not {"conversation_dir", "target_side", "mixture_start_seconds"} <= sample.keys():
        return None
    directory = Path(sample["conversation_dir"])
    side = "ab"[sample["target_side"]]
    target_words = transcript_words(directory / f"speaker_{side}_transcript.json")
    other_side = "b" if side == "a" else "a"
    other_words = transcript_words(directory / f"speaker_{other_side}_transcript.json")
    groups: list[tuple[float, float]] = []
    for start, end in target_words:
        if groups and start - groups[-1][1] <= 0.3:
            groups[-1] = (groups[-1][0], max(groups[-1][1], end))
        else:
            groups.append((start, end))
    crop_start = float(sample["mixture_start_seconds"])
    crop_end = crop_start + len(target) / SAMPLE_RATE
    mask = np.zeros(len(target), dtype=bool)
    count = 0
    for start, end in groups:
        if end - start > 1.2 or start < crop_start or end > crop_end:
            continue
        if not any(min(end, other_end) - max(start, other_start) >= 0.04
                   for other_start, other_end in other_words):
            continue
        low = max(0, round((start - crop_start) * SAMPLE_RATE))
        high = min(len(target), round((end - crop_start) * SAMPLE_RATE))
        mask[low:high] = True
        count += 1
    if not count or float(np.mean(other[mask] ** 2)) < 1e-8:
        return None
    before = float(np.mean(other[mask] ** 2))
    after = float(np.mean((estimate[mask] - target[mask]) ** 2))
    return 10 * math.log10((before + 1e-8) / (after + 1e-8)), count


def speaker_contrastive_loss(
    model: SpectralUNetExtractor,
    enrollment: torch.Tensor,
    target: torch.Tensor,
    other: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Make clean enrollment closer to the target than to the other speaker."""
    usable = (
        (enrollment.square().mean(-1) >= 1e-6)
        & (target.square().mean(-1) >= 1e-6)
        & (other.square().mean(-1) >= 1e-6)
    )
    if not usable.any():
        return target.new_zeros(()), 0
    voice, _ = model.encode_voice(enrollment[usable])
    same, _ = model.encode_voice(target[usable])
    different, _ = model.encode_voice(other[usable])
    logits = torch.stack(
        (F.cosine_similarity(voice, same), F.cosine_similarity(voice, different)), dim=-1
    ) / 0.1
    labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels), int(usable.sum())


def evaluate(model, dataset, device, demo_dir: Path, demo_count: int = 4) -> dict[str, float]:
    model.eval()
    scores: list[dict[str, float]] = []
    reverse_scores: list[float] = []
    enrollment_deltas: list[float] = []
    short_turn_scores: list[float] = []
    short_turn_count = 0
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
            short_turn = short_overlap_turn_score(sample, estimate, target, other)
            if short_turn is not None:
                short_turn_scores.append(short_turn[0])
                short_turn_count += short_turn[1]
            if (
                "other_enrollment" in sample
                and float(sample["other_enrollment"].square().mean()) >= 1e-6
            ):
                opposite = (
                    model(mixture, sample["other_enrollment"][None].to(device))
                    .waveform[0]
                    .cpu()
                    .numpy()
                )
                reverse = score_extraction(opposite, other, target, SAMPLE_RATE)
                if np.isfinite(reverse.sisdri_overlap):
                    reverse_scores.append(reverse.sisdri_overlap)
                enrollment_deltas.append(
                    float(np.sqrt(np.mean((estimate - opposite) ** 2)))
                    / max(float(np.sqrt(np.mean(estimate**2))), 1e-6)
                )
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
    summary = {
        key: float(np.mean(values))
        for key in scores[0]
        if (values := [score[key] for score in scores if np.isfinite(score[key])])
    }
    if reverse_scores:
        summary["reverse_sisdri_overlap"] = float(np.mean(reverse_scores))
    if enrollment_deltas:
        summary["enrollment_delta_ratio"] = float(np.mean(enrollment_deltas))
        summary["reverse_examples"] = float(len(enrollment_deltas))
    if short_turn_scores:
        summary["short_overlap_turn_improvement_db"] = float(np.mean(short_turn_scores))
        summary["short_overlap_turn_examples"] = float(len(short_turn_scores))
        summary["short_overlap_turn_count"] = float(short_turn_count)
    return summary


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
    train_pairs, val_pairs, test_pairs = select_splits(pairs, args.split_manifest, args.seed)
    if args.split_manifest is None:
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
    if args.init_model:
        with args.init_model.open("rb") as source:
            manifest["initial_model_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
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
        enrollment_selection=args.enrollment_selection,
    )
    if args.cross_pair_probability:
        train_set = CrossPairWindows(train_set, args.cross_pair_probability, args.seed + 2000)
    val_set = OpenYapWindows(
        val_pairs,
        args.val_examples,
        args.seed + 10_000,
        args.mixture_seconds,
        args.enrollment_seconds,
        enrollment_selection=args.enrollment_selection,
    )
    model_config = {"n_fft": args.n_fft, "hop_length": args.hop_length}
    if args.architecture == "band":
        model_config.update(
            bands=args.bands, width=args.width, blocks=args.blocks,
            conditioning=args.conditioning,
        )
        model = TargetSpeakerExtractor(**model_config).to(args.device)
    else:
        model = SpectralUNetExtractor(**model_config).to(args.device)
    if args.init_model:
        load_initial_model(model, args.init_model, args.architecture, model_config)
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
            parts["cross_pair_fraction"] = float(sample["cross_pair"].float().mean())
            if args.mask_supervision_weight:
                mask_loss = mask_supervision_loss(
                    output, target, other, args.n_fft, args.hop_length
                )
                loss = loss + args.mask_supervision_weight * mask_loss
                parts["mask_supervision_loss"] = float(mask_loss.detach())
            if args.paired_directions:
                opposite_enrollment = sample["other_enrollment"].to(
                    args.device, non_blocking=True
                )
                usable = opposite_enrollment.square().mean(-1) >= 1e-6
                if usable.any():
                    opposite_output = model(mixture[usable], opposite_enrollment[usable])
                    opposite_loss, opposite_parts = extraction_loss(
                        opposite_output, other[usable], target[usable]
                    )
                    loss = 0.5 * (loss + opposite_loss)
                    parts.update(
                        {f"opposite_{key}": value for key, value in opposite_parts.items()}
                    )
                    parts["paired_examples"] = int(usable.sum())
                else:
                    parts["paired_examples"] = 0
            if args.speaker_contrastive_weight:
                speaker_loss, speaker_examples = speaker_contrastive_loss(
                    model, enrollment, target, other
                )
                loss = loss + args.speaker_contrastive_weight * speaker_loss
                parts["speaker_contrastive_loss"] = float(speaker_loss.detach())
                parts["speaker_contrastive_examples"] = speaker_examples
            if args.source_discriminative_weight:
                source_loss, source_examples = source_discriminative_loss(
                    output.waveform, target, other
                )
                loss = loss + args.source_discriminative_weight * source_loss
                parts["source_discriminative_loss"] = float(source_loss.detach())
                parts["source_discriminative_examples"] = source_examples
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
                        "architecture": args.architecture,
                        "model_config": model_config,
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
