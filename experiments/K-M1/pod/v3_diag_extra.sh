#!/usr/bin/env bash
# Kezek v3 stage 1 diagnosis (K-V3-S1-DIAG), extra questions: the v3s1c model with packed solo
# audio, no training, full recordings: activity F1 with a 200 ms collar and where false positives
# fall (validation, clean), the train/val gap (30 train conversations, clean), and which part of
# the recording chain hurts (validation rendered with only the per-voice part, only the mixture part).
# Results to HF k-v3-diag/extra/. Every run logs to the Airon tracker.
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_diag_extra.sh > /data/v3_diag_extra.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export AIRON_EXPERIMENT="K-V3-S1-DIAG stage 1 diagnosis: where swaps come from (v3s1c, no training)"
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
python -c "import json; d = json.load(open('$split')); print('\n'.join('conv_' + c for c in d['val'] + d['test'] + d['train'][::31][:30]))" > /data/keep.txt
export KEEP=/data/keep.txt
grep -q "AUG READY" /data/get_aug.log 2>/dev/null || ( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
grep -q "parts done: 115" /data/get_data.log 2>/dev/null || \
  for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 6 >> /data/get_data.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
mkdir -p /data/dx
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --max-minutes 100000 --model /data/init/k-v3-v3s1c/best.pt"
$e --conversations 113 --tag _collar --out /data/dx/collar_clean.json > /data/dx/collar_clean.log 2>&1 &
$e --split train --conversations 30 --tag _train30 --out /data/dx/train30_clean.json > /data/dx/train30_clean.log 2>&1 &
wait
# a full rendered recording takes about 12 GB of RAM: one at a time
for part in sources finish; do
  $e --conversations 113 --aug-root /data/aug --render-part $part --tag _rec_$part \
    --out /data/dx/rec_${part}.json > /data/dx/rec_${part}.log 2>&1
done
for f in /data/dx/*.log; do
  grep -q SUMMARY "$f" && echo "$(basename "$f" .log) $(grep -h SUMMARY "$f" | sed 's/SUMMARY //')"
done > /data/dx/summary_extra.txt
python - <<PY
import glob, os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("/data/dx/*.json")) + ["/data/dx/summary_extra.txt"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-diag/extra/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 DIAG EXTRA DONE"
