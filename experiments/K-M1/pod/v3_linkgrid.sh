#!/usr/bin/env bash
# Kezek v3 stage 1: linking settings on the window cache of one model (no new identity pass),
# full recordings of the validation conversations, clean, word-gated labels; two at a time.
# Results to HF k-v3-linkgrid/$TAG/. Every run logs to the Airon tracker.
#   TAG=g03_s3k MODEL=/data/snap/g03_s3k.pt CACHE=/data/snap/cache_s3k \
#     nohup bash ~/ke-zek/experiments/K-M1/pod/v3_linkgrid.sh > /data/linkgrid.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export KEZEK_LABELS=words KEZEK_WORD_GAP=${WORD_GAP:-0.3}
export AIRON_EXPERIMENT="K-V3-S1W stage 1 on word-gated activity labels"
out=/data/linkgrid/$TAG
mkdir -p $out
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $repo/experiments/K-R1/split_round1.json --max-minutes 100000 --conversations 113 --model $MODEL --cache $CACHE"
run() {  # name flags...
  local n=$1
  shift
  $e "$@" --tag "_${TAG}_$n" --out $out/$n.json > $out/$n.log 2>&1
}
run embw8 --link-params '{"emb_w": 8}' &
run embw2 --link-params '{"emb_w": 2}' &
wait
run ovw3 --link-params '{"overlap_w": 3}' &
run cap20 --link-params '{"solo_cap": 20}' &
wait
run relink2 --relink '{"cut": 2.0}' &
run relink4 --relink '{"cut": 4.0}' &
wait
for f in $out/*.log; do
  grep -q SUMMARY "$f" && echo "$(basename "$f" .log) $(grep -h SUMMARY "$f" | sed 's/SUMMARY //')"
done > $out/summary.txt
python - <<PY
import glob, os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("$out/*.json")) + ["$out/summary.txt"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-linkgrid/$TAG/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "LINKGRID $TAG DONE"
