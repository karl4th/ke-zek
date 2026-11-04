"""Fine-tune a pretrained two-speaker separator on aligned Open Yap tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from scipy.signal import resample_poly
from train import git_state, select_splits, upload_checkpoint

from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapSeparationWindows

SOURCE_REPO = "speechbrain/sepformer-wsj02mix"


def _si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    projection = (
        (estimate * reference).sum(dim=-1, keepdim=True)
        / reference.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ) * reference
    noise = estimate - projection
    score = 10 * torch.log10(
        (projection.square().sum(dim=-1) + 1e-8)
        / (noise.square().sum(dim=-1) + 1e-8)
    )
    valid = reference.square().mean(dim=-1) >= 1e-6
    return torch.where(valid, score.clamp(-30, 30), 0.0)


def permutation_loss(
    estimates: torch.Tensor, first: torch.Tensor, second: torch.Tensor, mixture: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use a scale-aware waveform loss plus PIT SI-SDR, including quiet-speaker windows."""
    if estimates.shape != (len(first), first.shape[-1], 2):
        raise ValueError("Estimated source shape does not match the references")
    normalizer = mixture.square().mean(dim=-1).clamp_min(1e-6)

    def direction(left: torch.Tensor, right: torch.Tensor):
        mse = (
            (estimates[:, :, 0] - left).square().mean(dim=-1)
            + (estimates[:, :, 1] - right).square().mean(dim=-1)
        ) / normalizer
        sisdr = _si_sdr(estimates[:, :, 0], left) + _si_sdr(estimates[:, :, 1], right)
        return mse - 0.05 * sisdr, mse, sisdr * 0.5

    direct, direct_mse, direct_sisdr = direction(first, second)
    swapped, swapped_mse, swapped_sisdr = direction(second, first)
    use_direct = direct <= swapped
    loss = torch.where(use_direct, direct, swapped).mean()
    mse = torch.where(use_direct, direct_mse, swapped_mse).mean()
    sisdr = torch.where(use_direct, direct_sisdr, swapped_sisdr).mean()
    return loss, {"pit_mse": float(mse.detach()), "pit_sisdr": float(sisdr.detach())}


def evaluate(separator, dataset, device: str) -> dict[str, float]:
    separator.mods.eval()
    results = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            mixture = sample["mixture"][None].to(device)
            first = sample["first"].numpy()
            second = sample["second"].numpy()
            reduced = AF.resample(mixture, 16_000, 8_000)
            estimates = separator.separate_batch(reduced)[0].cpu().numpy()
            sources = [
                resample_poly(estimates[:, side], 2, 1).astype("float32")[: len(first)]
                for side in range(2)
            ]
            sources = [
                np.pad(source, (0, len(first) - len(source))) if len(source) < len(first)
                else source for source in sources
            ]
            reconstruction = sources[0] + sources[1]
            mixture_np = mixture[0].cpu().numpy()
            gain = float(
                np.dot(mixture_np, reconstruction)
                / max(np.dot(reconstruction, reconstruction), 1e-8)
            )
            sources = [source * gain for source in sources]
            direct_error = np.mean((sources[0] - first) ** 2 + (sources[1] - second) ** 2)
            swapped_error = np.mean((sources[1] - first) ** 2 + (sources[0] - second) ** 2)
            if swapped_error < direct_error:
                sources.reverse()
            results.append(score_extraction(sources[0], first, second, 16_000).as_dict())
            results.append(score_extraction(sources[1], second, first, 16_000).as_dict())
    separator.mods.train()
    return {
        f"oracle_{key}": float(np.mean(values))
        for key in results[0]
        if (values := [row[key] for row in results if np.isfinite(row[key])])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--stop-at", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--overlap-sampling", type=float, default=0.5)
    parser.add_argument("--val-examples", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="v2-sepformer-finetune")
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.val_examples, args.eval_every) < 1:
        parser.error("Steps, batch size, validation examples, and interval must be positive")
    if args.stop_at < 0 or args.stop_at > args.steps:
        parser.error("Invalid --stop-at")
    if not 0 <= args.overlap_sampling <= 1:
        parser.error("Invalid overlap sampling probability")
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        parser.error("AIRON_API_KEY and AIRON_URL are required")

    from huggingface_hub import HfApi
    from speechbrain.inference.separation import SepformerSeparation

    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("A paid run must use a clean commit")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    source_revision = HfApi().model_info(SOURCE_REPO).sha
    pairs = scan_conversations(args.data_root)
    train_pairs, val_pairs, test_pairs = select_splits(pairs, args.split_manifest, args.seed)
    splits = {
        name: [pair.conversation_id for pair in group]
        for name, group in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key not in {"stop_at", "resume"}
    }
    manifest = {
        "commit": commit,
        "config": config,
        "splits": splits,
        "source_repo": SOURCE_REPO,
        "source_revision": source_revision,
    }
    manifest_path = args.out / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        if previous["config"] != config or previous["splits"] != splits:
            raise RuntimeError("Resume settings or split changed")
        manifest = previous
        source_revision = previous["source_revision"]
    else:
        if manifest_path.exists():
            raise RuntimeError("Output already contains a manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2))

    separator = SepformerSeparation.from_hparams(
        source=SOURCE_REPO,
        revision=source_revision,
        savedir=str(args.out.parent / "pretrained" / "speechbrain_sepformer-wsj02mix"),
        run_opts={"device": args.device},
        freeze_params=False,
    )
    if int(separator.hparams.sample_rate) != 8_000:
        raise RuntimeError("Unexpected source model sample rate")
    optimizer = torch.optim.AdamW(separator.mods.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = min(100, max(1, args.steps // 20))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, args.steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    checkpoint_path = args.out / "last.pt"
    start_step = 0
    best_score = -float("inf")
    if args.resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        separator.mods.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        best_score = float(state["best_score"])
    end_step = args.stop_at or args.steps
    if start_step >= end_step:
        raise RuntimeError("Checkpoint is already at or past the requested stop step")

    train_set = OpenYapSeparationWindows(
        train_pairs, args.steps * args.batch_size, args.seed,
        overlap_sampling=args.overlap_sampling,
    )
    val_set = OpenYapSeparationWindows(val_pairs, args.val_examples, args.seed + 10_000)
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
    import airon

    tracker = airon.init(
        project="kezek",
        experiment="K-V2 recipe search on Open Yap",
        name=args.run_name,
        config={
            **config, "commit": commit, "source_revision": source_revision,
            "split_sha256": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
            "stop_at": args.stop_at,
        },
    )
    try:
        if start_step == 0:
            baseline = evaluate(separator, val_set, args.device)
            print(f"eval step 0: {json.dumps(baseline, sort_keys=True)}", flush=True)
            tracker.log({f"val/{key}": value for key, value in baseline.items()}, step=0)
            best_score = baseline.get("oracle_sisdri_overlap", -float("inf"))
            torch.save(
                {
                    "model": separator.mods.state_dict(),
                    "source_repo": SOURCE_REPO,
                    "source_revision": source_revision,
                    "step": 0,
                },
                args.out / "best_model.pt",
            )
            upload_checkpoint(args, [args.out / "best_model.pt", manifest_path])
        for step, sample in enumerate(loader, start_step + 1):
            mixture = sample["mixture"].to(args.device, non_blocking=True)
            first = sample["first"].to(args.device, non_blocking=True)
            second = sample["second"].to(args.device, non_blocking=True)
            reduced_mix = AF.resample(mixture, 16_000, 8_000)
            reduced_first = AF.resample(first, 16_000, 8_000)
            reduced_second = AF.resample(second, 16_000, 8_000)
            estimates = separator.separate_batch(reduced_mix)
            loss, parts = permutation_loss(
                estimates, reduced_first, reduced_second, reduced_mix
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(separator.mods.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if step == 1 or step % 10 == 0:
                metrics = {"train/loss": float(loss.detach())}
                metrics.update({f"train/{key}": value for key, value in parts.items()})
                tracker.log(metrics, step=step)
                print(f"step {step}: {metrics}", flush=True)
            if step % args.eval_every == 0 or step == end_step:
                metrics = evaluate(separator, val_set, args.device)
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                score = metrics.get("oracle_sisdri_overlap", -float("inf"))
                model_payload = {
                    "model": separator.mods.state_dict(), "source_repo": SOURCE_REPO,
                    "source_revision": source_revision, "step": step,
                }
                model_path = args.out / "model.pt"
                torch.save(model_payload, model_path)
                paths = [model_path, manifest_path]
                if score > best_score:
                    best_score = score
                    best_path = args.out / "best_model.pt"
                    torch.save(model_payload, best_path)
                    paths.append(best_path)
                torch.save(
                    {
                        "model": separator.mods.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_score": best_score,
                    }, checkpoint_path,
                )
                paths.append(checkpoint_path)
                upload_checkpoint(args, paths)
            if step == end_step:
                break
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
