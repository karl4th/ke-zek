"""Inspect paired conversation audio before building extraction examples."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ConversationPair:
    conversation_id: str
    speaker_a_id: str
    speaker_b_id: str
    speaker_a_path: Path
    speaker_b_path: Path
    sample_rate: int
    frames: int

    @property
    def speaker_ids(self) -> frozenset[str]:
        return frozenset((self.speaker_a_id, self.speaker_b_id))


def _audio_path(directory: Path, side: str) -> Path:
    paths = [
        p for suffix in ("wav", "flac") if (p := directory / f"speaker_{side}.{suffix}").is_file()
    ]
    if len(paths) != 1:
        raise ValueError(f"{directory}: expected exactly one audio file for speaker {side}")
    return paths[0]


def scan_conversations(root: str | Path) -> list[ConversationPair]:
    """Read archive metadata and reject missing, unmatched, or non-mono tracks."""
    base = Path(root)
    if (base / "conversations").is_dir():
        base /= "conversations"
    if not base.is_dir():
        raise FileNotFoundError(base)

    pairs = []
    for meta_path in sorted(base.glob("conv_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        directory = meta_path.parent
        a_path = _audio_path(directory, "a")
        b_path = _audio_path(directory, "b")
        a_info, b_info = sf.info(a_path), sf.info(b_path)
        if a_info.channels != 1 or b_info.channels != 1:
            raise ValueError(f"{directory}: both speaker tracks must be mono")
        if a_info.samplerate != b_info.samplerate or a_info.frames != b_info.frames:
            raise ValueError(f"{directory}: speaker tracks are not sample-aligned")
        a_id, b_id = str(meta["speaker_a_id"]), str(meta["speaker_b_id"])
        if not a_id or not b_id or a_id == b_id:
            raise ValueError(f"{directory}: invalid speaker identities")
        pairs.append(
            ConversationPair(
                conversation_id=str(meta["conversation_id"]),
                speaker_a_id=a_id,
                speaker_b_id=b_id,
                speaker_a_path=a_path,
                speaker_b_path=b_path,
                sample_rate=a_info.samplerate,
                frames=a_info.frames,
            )
        )
    if not pairs:
        raise ValueError(f"{base}: no conversation pairs found")
    return pairs


def check_speaker_disjoint(*splits: list[ConversationPair]) -> None:
    """Fail when the same person appears in more than one data split."""
    seen: dict[str, int] = {}
    for split_index, pairs in enumerate(splits):
        for pair in pairs:
            for speaker_id in pair.speaker_ids:
                if speaker_id in seen and seen[speaker_id] != split_index:
                    raise ValueError(f"speaker {speaker_id} occurs in multiple splits")
                seen[speaker_id] = split_index


def split_conversations(
    pairs: list[ConversationPair],
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[ConversationPair], list[ConversationPair], list[ConversationPair]]:
    """Split whole connected speaker groups into train, validation, and test."""
    if len(pairs) < 3 or len(fractions) != 3 or any(x <= 0 for x in fractions):
        raise ValueError("Three positive split fractions and at least three pairs are required")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")

    parent: dict[str, str] = {}

    def find(speaker: str) -> str:
        parent.setdefault(speaker, speaker)
        if parent[speaker] != speaker:
            parent[speaker] = find(parent[speaker])
        return parent[speaker]

    for pair in pairs:
        a, b = pair.speaker_a_id, pair.speaker_b_id
        parent[find(a)] = find(b)

    groups: dict[str, list[ConversationPair]] = defaultdict(list)
    for pair in pairs:
        groups[find(pair.speaker_a_id)].append(pair)
    components = list(groups.values())
    if len(components) < 3:
        raise ValueError("At least three disconnected speaker groups are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(components)
    splits: list[list[ConversationPair]] = [[], [], []]
    for index, group in enumerate(components):
        if index < 3:
            destination = index
        else:
            fill = [
                len(split) / (len(pairs) * fraction)
                for split, fraction in zip(splits, fractions, strict=True)
            ]
            destination = int(np.argmin(fill))
        splits[destination].extend(group)
    for split in splits:
        split.sort(key=lambda pair: pair.conversation_id)
    check_speaker_disjoint(*splits)
    return splits[0], splits[1], splits[2]
