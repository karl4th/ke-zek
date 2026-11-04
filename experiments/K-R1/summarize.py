"""Markdown table of long-form K-R1 results: one row per JSON written by evaluate_long.py.

    python experiments/K-R1/summarize.py /data/final/val
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
    ("overlap_sisdri", "overlap SI-SDRi, dB"),
    ("target_solo_sisdr", "target solo SI-SDR, dB"),
    ("target_solo_level_db", "target solo level, dB"),
    ("other_solo_leak_db", "other-solo leakage, dB"),
    ("both_silent_db", "both silent, dB"),
    ("backchannel_recall", "short replies kept"),
)


def cell(d: dict, key: str) -> str:
    v = d.get(key)
    if v is None or v != v:
        return "n/a"
    ci = d.get("ci", {}).get(key)
    text = f"{v:.2f}" if key != "backchannel_recall" else f"{100 * v:.0f}%"
    if ci and key != "backchannel_recall":
        text += f" [{ci[0]:.1f}, {ci[1]:.1f}]"
    elif ci:
        text += f" [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]"
    return text


def main():
    root = Path(sys.argv[1])
    rows = []
    for f in sorted(root.glob("*.json")):
        if f.stem.endswith("_items"):
            continue
        d = json.loads(f.read_text())
        rows.append((f.stem, d))
    if not rows:
        raise SystemExit(f"no results in {root}")
    n = rows[0][1].get("excerpts")
    print(f"{root.name}: {n} excerpts of {rows[0][1].get('excerpt_s')} s\n")
    print("| method | " + " | ".join(name for _, name in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for name, d in rows:
        print(f"| {name} | " + " | ".join(cell(d, k) for k, _ in COLUMNS) + " |")


if __name__ == "__main__":
    main()
