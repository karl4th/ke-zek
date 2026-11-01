"""Kezek: each speaker of a two-person mono recording on their own track.

Step 1 (this package now): given a mono mixture of two people talking and a short sample of
the target speaker's voice (enrollment), return mono audio that contains only that speaker.
Step 2 runs it once per speaker to build two tracks (two mono or stereo).
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
