"""Keep a pod run safe without any other machine: push checkpoints to Hugging Face, delete the pod at the end.

HF_TOKEN comes from the RunPod secret in the pod's env. The pod is deleted only after the final
files are confirmed on the Hub, so a failed upload costs money, not the model.

    python guard.py --root /root/d --run kezek-run-360 --repo manifestro/kezek-k-t0c
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request

from huggingface_hub import HfApi


def say(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def pod_env() -> dict[str, str]:
    """RunPod keeps RUNPOD_API_KEY and RUNPOD_POD_ID in the environment of the container's first process."""
    raw = open("/proc/1/environ", "rb").read().decode(errors="ignore")
    return {**os.environ, **dict(v.split("=", 1) for v in raw.split("\0") if "=" in v)}


def delete_pod() -> None:
    env = pod_env()
    say(f"deleting pod {env['RUNPOD_POD_ID']}")
    r = subprocess.run(["runpodctl", "remove", "pod", env["RUNPOD_POD_ID"]], env=env, capture_output=True, text=True)
    say(f"runpodctl exit {r.returncode}: {r.stdout.strip()[:200]} {r.stderr.strip()[:200]}")
    if r.returncode == 0:
        return
    query = 'mutation { podTerminate(input: {podId: "%s"}) }' % env["RUNPOD_POD_ID"]
    req = urllib.request.Request(
        "https://api.runpod.io/graphql", data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {env['RUNPOD_API_KEY']}", "Content-Type": "application/json"},
    )
    say(f"graphql: {urllib.request.urlopen(req, timeout=30).read()[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True, help="run folder inside --root")
    ap.add_argument("--repo", required=True, help="private model repo, owner/name")
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()

    run = f"{a.root}/{a.run}"
    archive = f"{a.root}/archive"
    api, repo, sent, archived = None, a.repo, None, not os.path.isdir(archive)

    def push(names: list[str]) -> None:
        for name in names:
            api.upload_file(path_or_fileobj=f"{run}/{name}", path_in_repo=name, repo_id=repo)

    while True:
        time.sleep(a.every)
        try:
            if api is None:
                hub = HfApi(token=pod_env()["HF_TOKEN"])
                hub.create_repo(repo, private=True, exist_ok=True)
                api = hub
                say(f"uploading to {repo}")
            if not archived:
                api.upload_folder(folder_path=archive, path_in_repo="archive", repo_id=repo)
                archived = True
                say("archive uploaded")

            log = open(f"{run}/log.txt", errors="ignore").read()
            if "TRAINING DONE" in log:
                names = ["last.pt", "best.pt", "kezek_tse.pt", "eval_log.json", "log.txt"]
                push(names)
                remote = {f.rfilename: f.size for f in api.repo_info(repo, files_metadata=True).siblings}
                if all(remote.get(n) == os.path.getsize(f"{run}/{n}") for n in names):
                    say("final files confirmed on the Hub")
                    delete_pod()
                    return
                say("final files do not match the Hub yet")
                continue

            stamp = os.path.getmtime(f"{run}/last.pt")
            if stamp != sent:
                time.sleep(20)  # the eval writes best.pt and the log around last.pt
                push(["last.pt", "best.pt", "eval_log.json", "log.txt"])
                sent = stamp
                say(f"checkpoint uploaded, log: {[x for x in log.splitlines() if x.startswith('EVAL')][-1][:40]}")
        except Exception as e:
            say(f"upload problem, will retry: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
