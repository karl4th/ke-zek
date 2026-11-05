"""Send finished long-form evaluations (evaluate_long.py JSON files) to Airon, one run each.

    python experiments/K-R1/backfill_long.py /data/final/val/*.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_long import log_result  # noqa: E402

for path in sys.argv[1:]:
    if path.endswith(("_items.json", "_runs.json")):
        continue
    log_result(path, json.load(open(path)))
    print("sent", path)
