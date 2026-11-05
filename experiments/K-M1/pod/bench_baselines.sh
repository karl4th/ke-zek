#!/usr/bin/env bash
# Earlier checkpoints on kezek-v1-bench (baselines for Kezek v3), logged to the Airon tracker.
#   bash experiments/K-M1/pod/bench_baselines.sh   (after setup.sh)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../.."
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-EVAL final evaluation vs DialogueSidon"
rev=d0fdd6d90d460430724788198518677793047d64
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import hf_hub_download as d, snapshot_download
t = os.environ["HF_TOKEN"]
snapshot_download("manifestro/kezek-v1-bench", repo_type="dataset", revision="$rev", local_dir="/data/bench", token=t)
d("manifestro/kezek-k-t0c", "kezek_tse.pt", revision="3593d4232487a443f0ee7f7aadb5a79f69ad51b7", local_dir="/data/init/k-t0c", token=t)
for f in ("r5/kezek_tse.pt", "k-r2/step2500/kezek_tse.pt", "k-r2/step5000/kezek_tse.pt",
          "k-r2/step7500/kezek_tse.pt", "k-m1/block1_wavlm/model.pt"):
    d("manifestro/kezek-k-r1", f, local_dir="/data/init", token=t)
PY
python experiments/K-M1/bench_eval.py --bench /data/bench --revision $rev --out /data/bench_baselines.json \
  --tse kt0c=/data/init/k-t0c/kezek_tse.pt:flip r5=/data/init/r5/kezek_tse.pt \
        kr2_2500=/data/init/k-r2/step2500/kezek_tse.pt kr2_5000=/data/init/k-r2/step5000/kezek_tse.pt \
        kr2_7500=/data/init/k-r2/step7500/kezek_tse.pt \
  --s1 km1_block1_wavlm=/data/init/k-m1/block1_wavlm/model.pt
