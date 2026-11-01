"""Kezek: fast speaker diarization for podcasts, built on Mimi latents.

Kezek ("кезек" in Kazakh: turn, one's turn to speak) finds who speaks when in long mono
recordings, at the 12.5 Hz Mimi frame rate, and exports pseudo-duplex (two-channel)
conversations for training full-duplex spoken dialogue models.
"""

from kezek.annotation import Segment, read_rttm, to_activity, to_segments, write_rttm
from kezek.frames import MIMI_FRAME_RATE_HZ, SCORING_FRAME_RATE_HZ
from kezek.powerset import Powerset

__version__ = "0.0.1"

__all__ = [
    "MIMI_FRAME_RATE_HZ",
    "SCORING_FRAME_RATE_HZ",
    "Powerset",
    "Segment",
    "read_rttm",
    "to_activity",
    "to_segments",
    "write_rttm",
]
