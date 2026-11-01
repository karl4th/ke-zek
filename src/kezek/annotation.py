"""Speaker annotations: segments, RTTM I/O and frame-level activity matrices."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kezek.frames import MIMI_FRAME_RATE_HZ, frame_centers, num_frames


@dataclass(frozen=True, order=True)
class Segment:
    """A stretch of speech by one speaker, in seconds."""

    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"segment ends before it starts: {self}")

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_rttm(path: str | Path) -> dict[str, list[Segment]]:
    """Read an RTTM file into ``{file_id: [Segment, ...]}``.

    Only ``SPEAKER`` lines are used. Columns: type, file, channel, onset, duration, ortho,
    stype, name, conf, slat.
    """
    result: dict[str, list[Segment]] = defaultdict(list)
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0].startswith((";", "#")) or fields[0] != "SPEAKER":
            continue
        if len(fields) < 8:
            raise ValueError(f"{path}:{lineno}: malformed RTTM line: {line!r}")
        file_id, onset, duration, speaker = fields[1], float(fields[3]), float(fields[4]), fields[7]
        result[file_id].append(Segment(onset, onset + duration, speaker))
    return {k: sorted(v) for k, v in result.items()}


def write_rttm(path: str | Path, annotations: dict[str, Sequence[Segment]]) -> None:
    """Write ``{file_id: segments}`` as RTTM."""
    lines = []
    for file_id, segments in annotations.items():
        for s in sorted(segments):
            lines.append(
                f"SPEAKER {file_id} 1 {s.start:.3f} {s.duration:.3f} "
                f"<NA> <NA> {s.speaker} <NA> <NA>"
            )
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def read_uem(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Read a UEM file (``file_id channel start end`` per line) into scored regions."""
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0].startswith((";", "#")):
            continue
        if len(fields) < 4:
            raise ValueError(f"{path}:{lineno}: malformed UEM line: {line!r}")
        result[fields[0]].append((float(fields[2]), float(fields[3])))
    return dict(result)


def speakers_of(segments: Iterable[Segment]) -> list[str]:
    """Sorted list of distinct speaker labels."""
    return sorted({s.speaker for s in segments})


def end_time(segments: Iterable[Segment]) -> float:
    return max((s.end for s in segments), default=0.0)


def to_activity(
    segments: Sequence[Segment],
    duration_s: float | None = None,
    frame_rate: float = MIMI_FRAME_RATE_HZ,
    speakers: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Rasterise segments into a boolean ``[num_speakers, num_frames]`` activity matrix.

    ``duration_s`` defaults to the end of the last segment. ``speakers`` fixes the row order;
    by default rows are the sorted speaker labels.
    """
    if duration_s is None:
        duration_s = end_time(segments)
    n = num_frames(duration_s, frame_rate)
    labels = list(speakers) if speakers is not None else speakers_of(segments)
    index = {spk: i for i, spk in enumerate(labels)}
    activity = np.zeros((len(labels), n), dtype=bool)
    centers = frame_centers(n, frame_rate)
    for s in segments:
        if s.speaker not in index:
            raise KeyError(f"speaker {s.speaker!r} not in the given speaker list")
        lo = np.searchsorted(centers, s.start, side="left")
        hi = np.searchsorted(centers, s.end, side="left")
        activity[index[s.speaker], lo:hi] = True
    return activity, labels


def to_segments(
    activity: np.ndarray,
    speakers: Sequence[str],
    frame_rate: float = MIMI_FRAME_RATE_HZ,
) -> list[Segment]:
    """Convert a boolean ``[num_speakers, num_frames]`` matrix back into segments."""
    if activity.ndim != 2 or activity.shape[0] != len(speakers):
        raise ValueError("activity must be [num_speakers, num_frames] matching `speakers`")
    segments = []
    for row, spk in zip(activity, speakers, strict=True):
        padded = np.concatenate([[False], row.astype(bool), [False]])
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for on, off in zip(changes[0::2], changes[1::2], strict=True):
            segments.append(Segment(on / frame_rate, off / frame_rate, spk))
    return sorted(segments)
