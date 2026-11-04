"""Adapt a pretrained separator to output the enrolled speaker on channel zero."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from evaluate_sepformer import SEPARATOR_REPO, SPEAKER_REPO, _speaker_embedding, trim_by_energy
from train import git_state, select_splits, short_overlap_turn_score
from train_sepformer import _si_sdr, align_sources_to_mixture

from kezek.conditioned_separator import ConditionedSeparator
from kezek.metrics import score_extraction
from kezek.open_yap import scan_conversations
from kezek.open_yap_training import OpenYapWindows


class PairedTargetWindows(torch.utils.data.Dataset):
    """Present both requested voices for each training mixture."""

    def __init__(self, natural: OpenYapWindows) -> None:
        self.natural = natural

    def __len__(self) -> int:
        return 2 * len(self.natural)

    def __getitem__(self, index: int) -> dict:
        sample = self.natural[index // 2]
        if index % 2:
            sample["target"], sample["other"] = sample["other"], sample["target"]
            sample["enrollment"], sample["other_enrollment"] = (
                sample["other_enrollment"], sample["enrollment"]
            )
            sample["target_side"] = 1 - sample["target_side"]
            sample["target_speaker_id"], sample["other_speaker_id"] = (
                sample["other_speaker_id"], sample["target_speaker_id"]
            )
        return sample


def voice_embeddings(classifier, enrollment: torch.Tensor, device: str) -> torch.Tensor:
    voices = []
    with torch.inference_mode():
        for sample in enrollment:
            audio = trim_by_energy(sample.detach().cpu().numpy())
            embedding = _speaker_embedding(classifier, audio, device)
            voices.append(
                embedding if embedding is not None else torch.zeros(192, device=device)
            )
    return torch.stack(voices).clone()


def direct_loss(
    estimates: torch.Tensor, target: torch.Tensor, other: torch.Tensor,
    mixture: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train ordered target and residue outputs, including target-absent windows."""
    if estimates.shape != (len(target), target.shape[-1], 2):
        raise ValueError("Estimates and references have different shapes")
    normalizer = mixture.square().mean(dim=-1).clamp_min(1e-6)
    mse = (
        (estimates[..., 0] - target).square().mean(dim=-1)
        + (estimates[..., 1] - other).square().mean(dim=-1)
    ) / normalizer
    sisdr = (
        _si_sdr(estimates[..., 0], target)
        + _si_sdr(estimates[..., 1], other)
    ) * 0.5
    loss = (mse - 0.1 * sisdr).mean()
    return loss, {
        "direct_mse": float(mse.mean().detach()),
        "direct_sisdr": float(sisdr.mean().detach()),
    }


def evaluate(model, classifier, dataset, device: str) -> tuple[dict, list[dict]]:
    model.eval()
    scores = []
    oracle_scores = []
    selected_correct = []
    voiced_correct = []
    overlap_correct = []
    short_scores = []
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            mixture = sample["mixture"][None].to(device)
            reduced = AF.resample(mixture, 16_000, 8_000)
            voice = voice_embeddings(classifier, sample["enrollment"][None], device)
            estimates, _ = align_sources_to_mixture(model(reduced, voice), reduced)
            upsampled = AF.resample(estimates.transpose(1, 2).contiguous(), 8_000, 16_000)
            target = sample["target"].numpy()
            other = sample["other"].numpy()
            source_scores = [
                score_extraction(
                    upsampled[0, side, : len(target)].cpu().numpy(), target, other, 16_000
                )
                for side in range(2)
            ]
            selected = source_scores[0]
            scores.append(selected.as_dict())
            oracle = int(np.argmax([item.sisdr for item in source_scores]))
            if np.isfinite(source_scores[oracle].sisdri_overlap):
                oracle_scores.append(source_scores[oracle].sisdri_overlap)
            hit = float(oracle == 0)
            selected_correct.append(hit)
            if float(np.sqrt(np.mean(target**2))) >= 1e-3:
                voiced_correct.append(hit)
                if selected.overlap_s >= 0.1:
                    overlap_correct.append(hit)
            estimate = upsampled[0, 0, : len(target)].cpu().numpy()
            short = short_overlap_turn_score(sample, estimate, target, other)
            if short is not None:
                short_scores.append(short[0])
            rows.append({
                "conversation_id": sample["conversation_id"],
                "target_side": sample["target_side"],
                "speaker_selected_correct": bool(hit),
                "sisdri_overlap": (
                    selected.sisdri_overlap if np.isfinite(selected.sisdri_overlap) else None
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
    metrics.update({
        "oracle_sisdri_overlap": float(np.mean(oracle_scores)),
        "speaker_assignment_accuracy": float(np.mean(selected_correct)),
        "voiced_speaker_assignment_accuracy": float(np.mean(voiced_correct)),
        "overlap_speaker_assignment_accuracy": float(np.mean(overlap_correct)),
        "short_overlap_turn_improvement_db": float(np.mean(short_scores)),
        "short_overlap_turn_examples": float(len(short_scores)),
    })
    model.train()
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-examples", type=int, default=128)
    parser.add_argument("--enrollment-seconds", type=float, default=15.0)
    parser.add_argument("--unfreeze-masknet", action="store_true")
    parser.add_argument("--run-name", default="v2-conditioned-separator")
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
    backbone = SepformerSeparation.from_hparams(
        source=SEPARATOR_REPO,
        savedir=str(args.out.parent / "pretrained" / SEPARATOR_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    backbone.mods.load_state_dict(state["model"])
    classifier = EncoderClassifier.from_hparams(
        source=SPEAKER_REPO,
        savedir=str(args.out.parent / "pretrained" / SPEAKER_REPO.replace("/", "_")),
        run_opts={"device": args.device},
    )
    classifier.mods.eval()
    model = ConditionedSeparator(
        backbone.mods.encoder, backbone.mods.masknet, backbone.mods.decoder
    ).to(args.device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.masknet.parameters():
        parameter.requires_grad_(args.unfreeze_masknet)
    groups = [{"params": model.condition.parameters(), "lr": args.lr}]
    if args.unfreeze_masknet:
        groups.append({"params": model.masknet.parameters(), "lr": args.lr * 0.02})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    train_pairs, val_pairs, _ = select_splits(
        scan_conversations(args.data_root), args.split_manifest, 42
    )
    natural = OpenYapWindows(
        train_pairs, (args.steps * args.batch_size + 1) // 2, seed=27_904,
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
        "separator_step": state["step"],
        "separator_sha256": checksum.hexdigest(),
        "separator_revision": state["source_revision"],
        "speaker_revision": HfApi().model_info(SPEAKER_REPO).sha,
        "split_manifest": str(args.split_manifest),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "unfreeze_masknet": args.unfreeze_masknet,
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
        baseline, rows = evaluate(model, classifier, val_set, args.device)
        print(f"eval step 0: {json.dumps(baseline, sort_keys=True)}", flush=True)
        tracker.log({f"val/{key}": value for key, value in baseline.items()}, step=0)
        (args.out / "validation_000000.json").write_text(
            json.dumps({"metrics": baseline, "rows": rows}, indent=2)
        )
        for step, sample in enumerate(loader, 1):
            mixture = sample["mixture"].to(args.device)
            target = sample["target"].to(args.device)
            other = sample["other"].to(args.device)
            reduced_mix = AF.resample(mixture, 16_000, 8_000)
            reduced_target = AF.resample(target, 16_000, 8_000)
            reduced_other = AF.resample(other, 16_000, 8_000)
            voice = voice_embeddings(classifier, sample["enrollment"], args.device)
            estimates, _ = align_sources_to_mixture(
                model(reduced_mix, voice), reduced_mix
            )
            loss, measures = direct_loss(
                estimates, reduced_target, reduced_other, reduced_mix
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite conditioned loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 20 == 0:
                payload = {"train/loss": float(loss.detach())}
                payload.update({f"train/{key}": value for key, value in measures.items()})
                tracker.log(payload, step=step)
                print(f"step {step}: {payload}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, rows = evaluate(model, classifier, val_set, args.device)
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                tracker.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                torch.save({
                    "model": model.state_dict(), "step": step, "config": config,
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
