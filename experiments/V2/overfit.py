"""Check whether the extractor can memorize fixed, real paired-track windows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from train import git_state

from kezek.metrics import regions, score_extraction
from kezek.open_yap import scan_conversations, split_conversations
from kezek.open_yap_training import SAMPLE_RATE, OpenYapWindows
from kezek.tse_v2 import TargetSpeakerExtractor
from kezek.tse_v2_loss import extraction_loss


def select_windows(pairs, seed: int, count: int = 4) -> tuple[list[int], list[dict]]:
    candidates = OpenYapWindows(pairs, 512, seed)
    selected: list[int] = []
    examples: list[dict] = []
    for index in range(len(candidates)):
        if sum(i % 2 == index % 2 for i in selected) >= count // 2:
            continue
        sample = candidates[index]
        reg = regions(sample["target"].numpy(), sample["other"].numpy(), SAMPLE_RATE)
        if reg.overlap.sum() < 15 or reg.solo.sum() < 5 or reg.absent.sum() < 5:
            continue
        if float(sample["enrollment"].square().mean()) < 1e-6:
            continue
        selected.append(index)
        examples.append(sample)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} suitable fixed windows were found")
    return selected, examples


def evaluate_fixed(model, examples, device) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        mixture = torch.stack([sample["mixture"] for sample in examples]).to(device)
        enrollment = torch.stack([sample["enrollment"] for sample in examples]).to(device)
        estimate = model(mixture, enrollment).waveform
        switched = model(mixture, enrollment.roll(1, 0)).waveform
        swap_delta = (estimate - switched).square().mean(dim=-1).sqrt()
        estimate_rms = estimate.square().mean(dim=-1).sqrt().clamp_min(1e-8)
        scores = [
            score_extraction(
                estimate[index].cpu().numpy(),
                sample["target"].numpy(),
                sample["other"].numpy(),
                SAMPLE_RATE,
            ).as_dict()
            for index, sample in enumerate(examples)
        ]
    model.train()
    result = {
        key: float(np.mean(values))
        for key in scores[0]
        if (values := [score[key] for score in scores if np.isfinite(score[key])])
    }
    result["enrollment_swap_delta_ratio"] = float((swap_delta / estimate_rms).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--run-name", default="kezek-v2-fixed-overfit")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.lr <= 0:
        parser.error("Steps, evaluation interval, and learning rate must be positive")
    if not os.environ.get("AIRON_API_KEY") or not os.environ.get("AIRON_URL"):
        raise RuntimeError("AIRON_API_KEY and AIRON_URL are required")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("A paid run must use a clean, committed repository")

    import airon
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    if not api.repo_info(args.hf_repo).private:
        raise RuntimeError("Checkpoint repository must be private")
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    train, val, test = split_conversations(scan_conversations(args.data_root), seed=args.seed)
    indices, examples = select_windows(train, args.seed)
    manifest = {
        "commit": commit,
        "selected_indices": indices,
        "train_conversations": len(train),
        "val_conversations": len(val),
        "test_conversations": len(test),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    model = TargetSpeakerExtractor().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    mixture = torch.stack([sample["mixture"] for sample in examples]).to(args.device)
    enrollment = torch.stack([sample["enrollment"] for sample in examples]).to(args.device)
    target = torch.stack([sample["target"] for sample in examples]).to(args.device)
    other = torch.stack([sample["other"] for sample in examples]).to(args.device)
    run = airon.init(
        project="kezek",
        name=args.run_name,
        experiment="K-V2 recipe search on Open Yap",
        config=manifest,
        tags=["v2", "open-yap", "fixed-overfit"],
        total_steps=args.steps,
    )
    try:
        for step in range(args.steps + 1):
            if step == 0 or step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate_fixed(model, examples, args.device)
                print(f"eval step {step}: {json.dumps(metrics, sort_keys=True)}", flush=True)
                run.log({f"fixed/{key}": value for key, value in metrics.items()}, step=step)
                checkpoint = args.out / "model.pt"
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": {
                            "n_fft": model.n_fft,
                            "hop_length": model.hop_length,
                            "bands": len(model.band_sizes),
                            "width": 128,
                            "blocks": len(model.separator),
                        },
                        "step": step,
                    },
                    temporary,
                )
                os.replace(temporary, checkpoint)
                for path in (checkpoint, manifest_path):
                    api.upload_file(
                        path_or_fileobj=path,
                        repo_id=args.hf_repo,
                        path_in_repo=f"{args.run_name}/{path.name}",
                        commit_message=f"Save {args.run_name} step {step}",
                    )
            if step == args.steps:
                break
            output = model(mixture, enrollment)
            loss, parts = extraction_loss(output, target, other)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 50 == 0:
                print(f"step {step + 1}: loss={float(loss.detach()):.3f}", flush=True)
                run.log(
                    {
                        "train/loss": float(loss.detach()),
                        **{f"train/{k}": v for k, v in parts.items()},
                    },
                    step=step + 1,
                )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
