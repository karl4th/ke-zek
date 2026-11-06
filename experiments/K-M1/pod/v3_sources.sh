#!/usr/bin/env bash
# Kezek v3 step 1b: where do the swapped blocks come from (s1_swap_sources.py). The v3s1c model
# with packed solo audio (pretrained ECAPA) on full held-out conversations, clean and rendered,
# with a window cache, then the swap sources from that cache; results to HF k-v3-sources/.
# Every run logs to the Airon tracker.
#   git clone -b kezek-r1 https://github.com/karl4th/ke-zek.git ~/ke-zek
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_sources.sh > /data/v3_sources.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-S1 stage 1: profiles and activity (WavLM + Conformer, 16 s)"
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import hf_hub_download as d
d("manifestro/kezek-k-r1", "k-v3-v3s1c/best.pt", local_dir="/data/init", token=os.environ["HF_TOKEN"])
print("MODELS READY")
PY
split=$repo/experiments/K-R1/split_round1.json
python -c "import json; d = json.load(open('$split')); print('\n'.join('conv_' + c for c in d['val'] + d['test']))" > /data/keep.txt
export KEEP=/data/keep.txt
grep -q "AUG READY" /data/get_aug.log 2>/dev/null || ( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 6 >> /data/get_data.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
model=/data/init/k-v3-v3s1c/best.pt
mkdir -p /data/src
export KEZEK_PACK_SOLO=1
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --conversations 113 --model $model --cache /data/src/cache"
$e --tag _sources --out /data/src/base_clean.json > /data/src/base_clean.log 2>&1 &
$e --tag _sources --aug-root /data/aug --out /data/src/base_rec.json > /data/src/base_rec.log 2>&1 &
wait
for r in "" "--rec"; do
  python experiments/K-M1/s1_swap_sources.py --data /data/oy16 --split-file $split --cache /data/src/cache $r \
    --out /data/src/sources${r:+_rec}.json > /data/src/sources${r:+_rec}.log 2>&1
done
grep -H SUMMARY /data/src/*.log | sed 's/SUMMARY //' > /data/src/summary.txt
tail -qn 1 /data/src/sources*.log >> /data/src/summary.txt
python - <<PY
import glob, os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("/data/src/*.json")) + ["/data/src/summary.txt"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-sources/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 SOURCES DONE"
