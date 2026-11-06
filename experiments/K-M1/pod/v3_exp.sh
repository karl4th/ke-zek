#!/usr/bin/env bash
# Kezek v3 step 1b experiments on the window cache of v3_sources.sh (same instance): each variant
# on full held-out conversations, clean and rendered, two variants at a time; results to HF
# k-v3-exp/. Every run logs to the Airon tracker.
#   bash ~/ke-zek/experiments/K-M1/pod/v3_exp.sh name='--flags' [name='--flags' ...]
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export AIRON_EXPERIMENT="K-V3-S1 stage 1: profiles and activity (WavLM + Conformer, 16 s)"
split=$repo/experiments/K-R1/split_round1.json
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --conversations 113 --model /data/init/k-v3-v3s1c/best.pt --cache /data/src/cache"
mkdir -p /data/exp
n=0
for spec in "$@"; do
  name=${spec%%=*}
  flags=${spec#*=}
  $e $flags --tag "_$name" --out /data/exp/${name}_clean.json > /data/exp/${name}_clean.log 2>&1 &
  $e $flags --tag "_$name" --aug-root /data/aug --out /data/exp/${name}_rec.json > /data/exp/${name}_rec.log 2>&1 &
  n=$((n + 1))
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
for spec in "$@"; do
  name=${spec%%=*}
  for r in clean rec; do echo "$name $r $(grep -h SUMMARY /data/exp/${name}_$r.log | sed 's/SUMMARY //')"; done
done >> /data/exp/summary.txt
python - <<PY
import glob, os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("/data/exp/*.json")) + ["/data/exp/summary.txt"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-exp/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 EXP DONE"
