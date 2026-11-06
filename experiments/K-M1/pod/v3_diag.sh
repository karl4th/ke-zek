#!/usr/bin/env bash
# Kezek v3 stage 1 diagnosis (K-V3-S1-DIAG): the v3s1c model with packed solo audio, no training,
# full recordings of the 113 validation conversations, clean and rendered.
#   main: predicted activity, solo threshold 0.5, hop 8 s (window cache), then on that cache
#         predicted activity with solo threshold 0.9 and labelled activity (oracle)
#   hop2: predicted activity, solo threshold 0.5, hop 2 s (timed)
# Results to HF k-v3-diag/. Every run logs to the Airon tracker.
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_diag.sh main|hop2 > /data/v3_diag.log 2>&1 &
set -uo pipefail
mode=$1
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
python -c "import json; d = json.load(open('$split')); print('\n'.join('conv_' + c for c in d['val'] + d['test']))" > /data/keep.txt
export KEEP=/data/keep.txt
grep -q "AUG READY" /data/get_aug.log 2>/dev/null || ( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
grep -q "parts done: 115" /data/get_data.log 2>/dev/null || \
  for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 6 >> /data/get_data.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
mkdir -p /data/diag
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --conversations 113 --max-minutes 100000 --model /data/init/k-v3-v3s1c/best.pt"
# a full rendered recording (up to 76 min) takes about 12 GB of RAM to render: rendered runs go
# one at a time and next to clean runs only
cl() { local n=$1; shift; $e "$@" --tag "_$n" --out /data/diag/${n}_clean.json > /data/diag/${n}_clean.log 2>&1; }
rc() { local n=$1; shift; $e "$@" --tag "_$n" --aug-root /data/aug --out /data/diag/${n}_rec.json > /data/diag/${n}_rec.log 2>&1; }
if [ "$mode" = main ]; then
  c="--cache /data/diag/cache_h8"
  cl pred05 $c
  cl label $c --reembed label_full &
  cl pred09 $c --reembed pred --solo-thr 0.9 &
  wait
  rc pred05 $c
  rc label $c --reembed label_full
  rc pred09 $c --reembed pred --solo-thr 0.9
else
  cl pred05_hop2 --hop-s 2 --cache /data/diag/cache_h2
  rc pred05_hop2 --hop-s 2 --cache /data/diag/cache_h2
fi
for f in /data/diag/*.log; do
  grep -q SUMMARY "$f" && echo "$(basename "$f" .log) $(grep -h SUMMARY "$f" | sed 's/SUMMARY //')"
done > /data/diag/summary_$mode.txt
python - <<PY
import glob, os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("/data/diag/*.json")) + ["/data/diag/summary_$mode.txt"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-diag/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 DIAG $mode DONE"
