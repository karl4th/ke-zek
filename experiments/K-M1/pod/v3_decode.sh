#!/usr/bin/env bash
# Kezek v3 stage 1, activity decoding without retraining (K-V3-S1-DIAG follow-up): the v3s1c
# model with packed solo audio on full recordings of the validation conversations, clean:
#   overlap0.8, overlap0.9: marginals with an overlap kept only where the weaker slot is above
#     0.8 / 0.9 (window cache of v3_diag_extra.sh);
#   powerset: the most likely powerset class per frame (new identity pass that keeps the classes).
# Runs on the instance of v3_diag_extra.sh once its clean runs are done, one at a time next to
# its rendered runs (RAM). Results to HF
# k-v3-diag/decode/. Every run logs to the Airon tracker.
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_decode.sh > /data/v3_decode.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export AIRON_EXPERIMENT="K-V3-S1-DIAG stage 1 diagnosis: where swaps come from (v3s1c, no training)"
until grep -qs SUMMARY /data/dx/train30_clean.log || grep -qs Traceback /data/dx/train30_clean.log; do sleep 60; done
split=$repo/experiments/K-R1/split_round1.json
mkdir -p /data/dd
v="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --max-minutes 100000 --conversations 113 --model /data/init/k-v3-v3s1c/best.pt"
$v --cache /data/dd/cache_ps --decode powerset --tag _powerset --out /data/dd/powerset_clean.json > /data/dd/powerset_clean.log 2>&1
for t in 0.8 0.9; do
  $v --cache /data/dx/cache_h8 --decode overlap --overlap-thr $t --tag _overlap$t --out /data/dd/overlap${t}_clean.json > /data/dd/overlap${t}_clean.log 2>&1
done
for f in /data/dd/*.log; do
  grep -q SUMMARY "$f" && echo "$(basename "$f" .log) $(grep -h SUMMARY "$f" | sed 's/SUMMARY //')"
done > /data/dd/summary_decode.txt
python - <<PY
import glob, os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("/data/dd/*.json")) + ["/data/dd/summary_decode.txt"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-diag/decode/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 DECODE DONE"
