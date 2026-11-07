"""Merge the shards of one ``s1_eval_full.py --shard k --shards n`` evaluation into one result
(same metrics as an unsharded run) and log it to the tracker.

    python experiments/K-M1/merge_shards.py --out merged.json --name s1_full_val_x shard0.json shard1.json ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "K-R1"))
import tracker  # noqa: E402
from s1_eval_full import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="tracker run name")
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    parts = [json.load(open(f)) for f in a.shards]
    per_conv = [p for d in parts for p in d["per_conversation"]]
    rows = [r for d in parts for r in d["windows"]]
    runs = [r for d in parts for r in d["swap_runs"]]
    hours = sum(d.get("hours", 0.0) for d in parts)
    out = summarize(per_conv, rows, runs, hours, sum(d.get("t_ident", 0.0) for d in parts), sum(d.get("h_ident", 0.0) for d in parts))
    json.dump({"summary": out, "per_conversation": per_conv, "windows": rows, "swap_runs": runs,
               "shards": a.shards, "args": parts[0]["args"]}, open(a.out, "w"), indent=1)
    print("SUMMARY", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}), flush=True)
    run = tracker.init(a.name, {"shards": a.shards, **parts[0]["args"]}, ["kezek-v3", "stage1", "full-recording", "merged"], 1)
    sfx = "_rec" if parts[0]["args"].get("aug_root") else ""
    tracker.log(run, {f"full_val{sfx}/{k}": v for k, v in out.items() if isinstance(v, (int, float))}, len(per_conv))
    tracker.finish(run)


if __name__ == "__main__":
    main()
