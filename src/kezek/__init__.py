"""Kezek: extract one speaker's voice from a two-speaker mono recording.

Given a mono mixture of two people talking and a short sample of the target speaker's voice
(enrollment), Kezek returns mono audio that contains only the target speaker.
"""

from kezek.enrollment import pick_enrollment
from kezek.metrics import ExtractionScores, regions, score_extraction, si_sdr
from kezek.mixing import Placement, conversation_schedule, mix_pair, render

__version__ = "0.0.1"

__all__ = [
    "ExtractionScores",
    "Placement",
    "conversation_schedule",
    "mix_pair",
    "pick_enrollment",
    "regions",
    "render",
    "score_extraction",
    "si_sdr",
]
