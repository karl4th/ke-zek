"""Airon run tracking for K-R1, if the client and the key are there; otherwise a no-op.

On Airon AWS instances the key is in /etc/airon.env (AIRON_URL, AIRON_API_KEY, HF_TOKEN). It is
read into the environment here, never printed.
"""

from __future__ import annotations

import os

ENV_FILE = "/etc/airon.env"


def load_env(path: str = ENV_FILE) -> None:
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def init(name: str, config: dict, tags: list[str], total_steps: int):
    load_env()
    if not os.environ.get("AIRON_API_KEY"):
        print("airon tracking is off: no AIRON_API_KEY", flush=True)
        return None
    try:
        import airon

        return airon.init(
            project="kezek",
            name=name,
            experiment="K-R1",
            config=config,
            tags=["K-R1", *tags],
            total_steps=total_steps,
        )
    except Exception as e:
        print(f"airon tracking is off: {type(e).__name__}: {e}", flush=True)
        return None


def log(run, values: dict, step: int) -> None:
    if run is None:
        return
    clean = {k: float(v) for k, v in values.items() if isinstance(v, (int, float)) and v == v}
    try:
        run.log(clean, step=step)
    except Exception as e:
        print(f"airon log failed: {type(e).__name__}: {e}", flush=True)


def finish(run) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception as e:
            print(f"airon finish failed: {type(e).__name__}: {e}", flush=True)
