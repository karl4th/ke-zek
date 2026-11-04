"""Preserve one historical separator checkpoint under a step-specific private Hub path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.step < 0:
        parser.error("Step must be nonnegative")
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN is required")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    if not api.repo_info(args.repo).private:
        raise RuntimeError("Checkpoint repository must be private")
    filename = f"{args.prefix}/best_model.pt"
    match = None
    for commit in api.list_repo_commits(args.repo):
        if commit.title != f"Save {args.prefix} best_model.pt":
            continue
        candidate = Path(hf_hub_download(
            args.repo, filename, revision=commit.commit_id,
            token=token, cache_dir=str(args.out_dir / "cache"),
        ))
        state = torch.load(candidate, map_location="cpu", weights_only=True)
        if state.get("step") == args.step:
            match = (commit.commit_id, candidate, state)
            break
    if match is None:
        raise RuntimeError(f"No historical best_model.pt at step {args.step}")
    revision, source, state = match
    checksum = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    destination = f"{args.prefix}/step_{args.step:06d}/best_model.pt"
    manifest = {
        "step": args.step,
        "source_commit": revision,
        "source_repo": state["source_repo"],
        "source_revision": state["source_revision"],
        "sha256": checksum.hexdigest(),
    }
    manifest_path = args.out_dir / f"step_{args.step:06d}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    api.upload_file(
        path_or_fileobj=source,
        path_in_repo=destination,
        repo_id=args.repo,
        commit_message=f"Preserve {args.prefix} step {args.step}",
    )
    api.upload_file(
        path_or_fileobj=manifest_path,
        path_in_repo=f"{args.prefix}/step_{args.step:06d}/manifest.json",
        repo_id=args.repo,
        commit_message=f"Record {args.prefix} step {args.step} checksum",
    )
    remote = {
        item.rfilename: item.size
        for item in api.repo_info(args.repo, files_metadata=True).siblings
    }
    if remote.get(destination) != source.stat().st_size:
        raise RuntimeError("Archived checkpoint size does not match source")
    print(json.dumps({"repo": args.repo, "path": destination, **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
