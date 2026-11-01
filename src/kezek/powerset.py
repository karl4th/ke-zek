"""Powerset encoding of local speaker activity (pyannote-style, arXiv 2310.13025).

Within a window, at most ``max_speakers`` local speakers are tracked and at most
``max_overlap`` of them may speak at the same time. Each frame is assigned one class:
the empty set (non-speech), a single speaker, or a set of simultaneous speakers. A single
softmax over these classes gives voice activity, speaker identity and overlap at once.

With ``max_speakers=3, max_overlap=2`` the classes are
``{}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}`` (7 classes).
"""

from __future__ import annotations

from functools import cached_property
from itertools import combinations

import numpy as np

IGNORE_INDEX = -100
"""Target value for frames that must not contribute to the loss (PyTorch convention)."""


class Powerset:
    def __init__(self, max_speakers: int = 3, max_overlap: int = 2) -> None:
        if max_speakers < 1 or not 1 <= max_overlap <= max_speakers:
            raise ValueError("need max_speakers >= 1 and 1 <= max_overlap <= max_speakers")
        self.max_speakers = max_speakers
        self.max_overlap = max_overlap

    @cached_property
    def classes(self) -> list[tuple[int, ...]]:
        """Speaker sets, ordered by size then lexicographically; class 0 is non-speech."""
        return [
            combo
            for k in range(self.max_overlap + 1)
            for combo in combinations(range(self.max_speakers), k)
        ]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @cached_property
    def mapping(self) -> np.ndarray:
        """``[num_classes, max_speakers]`` 0/1 matrix: which speakers are active in each class."""
        m = np.zeros((self.num_classes, self.max_speakers), dtype=np.int8)
        for c, speakers in enumerate(self.classes):
            m[c, list(speakers)] = 1
        return m

    def encode(self, activity: np.ndarray, strict: bool = False) -> np.ndarray:
        """Map a ``[max_speakers, T]`` activity matrix to ``[T]`` class ids.

        Frames with more than ``max_overlap`` active speakers get ``IGNORE_INDEX``
        (or raise if ``strict``).
        """
        activity = np.asarray(activity, dtype=bool)
        if activity.ndim != 2 or activity.shape[0] != self.max_speakers:
            raise ValueError(f"expected [{self.max_speakers}, T] activity, got {activity.shape}")
        weights = 1 << np.arange(self.max_speakers)
        codes = weights @ activity.astype(np.int64)
        lookup = np.full(1 << self.max_speakers, IGNORE_INDEX, dtype=np.int64)
        for c, speakers in enumerate(self.classes):
            lookup[sum(1 << s for s in speakers)] = c
        ids = lookup[codes]
        if strict and np.any(ids == IGNORE_INDEX):
            raise ValueError(f"some frames have more than {self.max_overlap} active speakers")
        return ids

    def decode(self, ids: np.ndarray) -> np.ndarray:
        """Map ``[T]`` class ids back to a boolean ``[max_speakers, T]`` activity matrix.

        Frames marked ``IGNORE_INDEX`` decode to all-inactive.
        """
        ids = np.asarray(ids)
        out = np.zeros((self.max_speakers, ids.shape[0]), dtype=bool)
        valid = ids != IGNORE_INDEX
        out[:, valid] = self.mapping[ids[valid]].T.astype(bool)
        return out

    def to_multilabel(self, probs: np.ndarray) -> np.ndarray:
        """Turn ``[..., num_classes]`` class probabilities into ``[..., max_speakers]``
        per-speaker probabilities (marginals)."""
        return np.asarray(probs) @ self.mapping
