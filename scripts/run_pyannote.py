"""Run a pyannote diarization pipeline over a list of audio files and write RTTM (K0 baseline).

    pip install -e ".[baselines]"
    export HF_TOKEN=...   # the pyannote models are gated on Hugging Face
    python scripts/run_pyannote.py --audio-list data/k0_audio.txt --out runs/k0/pyannote31.rttm

``--audio-list`` holds one audio path per line; the file stem becomes the RTTM file id.
Also writes ``<out>.speed.json`` with the real-time factor, for the speed table of the paper.

Status: written for K0, not yet run against a real episode. Check the output of the first
episode by hand before scoring the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def load_pipeline(name: str, device: str):
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    try:  # pyannote.audio >= 4
        pipeline = Pipeline.from_pretrained(name, token=token)
    except TypeError:  # pyannote.audio 3.x
        pipeline = Pipeline.from_pretrained(name, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(f"could not load {name}; accept its terms on Hugging Face first")
    pipeline.to(torch.device(device))
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audio-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pipeline", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-speakers", type=int, default=None)
    args = parser.parse_args()

    import soundfile as sf

    pipeline = load_pipeline(args.pipeline, args.device)
    paths = [Path(p) for p in args.audio_list.read_text().split() if p]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    audio_s = wall_s = 0.0
    with args.out.open("w") as rttm:
        for path in paths:
            audio_s += sf.info(str(path)).duration
            t0 = time.perf_counter()
            kwargs = {"num_speakers": args.num_speakers} if args.num_speakers else {}
            output = pipeline(str(path), **kwargs)
            wall_s += time.perf_counter() - t0
            # pyannote.audio 4 returns an object with .speaker_diarization; 3.x an Annotation.
            annotation = getattr(output, "speaker_diarization", output)
            annotation.uri = path.stem
            annotation.write_rttm(rttm)
            print(f"{path.stem}: done", flush=True)

    speed = {
        "pipeline": args.pipeline,
        "device": args.device,
        "audio_hours": audio_s / 3600,
        "wall_hours": wall_s / 3600,
        "x_real_time": audio_s / wall_s if wall_s else None,
    }
    Path(f"{args.out}.speed.json").write_text(json.dumps(speed, indent=2))
    print(json.dumps(speed, indent=2))


if __name__ == "__main__":
    main()
