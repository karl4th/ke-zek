"""Send the validation curve of a finished K-R1 run (its eval_log.json) to Airon as a run.

    python experiments/K-R1/backfill.py /data/runs/r1_kt0c_state --name r1_kt0c_state
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="*", default=["backfill"])
    a = ap.parse_args()
    log = json.load(open(os.path.join(a.run_dir, "eval_log.json")))
    manifest_path = os.path.join(a.run_dir, "manifest.json")
    config = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    steps = config.get("args", {}).get("steps") or max(e["step"] for e in log)
    run = tracker.init(a.name, config, a.tags, steps)
    if run is None:
        raise SystemExit("no tracker")
    for e in log:
        tracker.log(run, {f"val/{k}": v for k, v in e.items() if k != "step"}, e["step"])
    tracker.finish(run)
    print(f"sent {len(log)} evaluations of {a.run_dir}")


if __name__ == "__main__":
    main()
