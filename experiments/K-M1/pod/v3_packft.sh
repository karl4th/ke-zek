#!/usr/bin/env bash
# Kezek v3 step 1b, part C on its own: the v3s1c model with packed solo audio and the fine-tuned
# ECAPA (HF k-v3-ecapa1/ecapa.pt) on full held-out conversations, clean and rendered, link
# diagnosis and kezek-v1-bench; results to HF k-v3-ecapa1/. Every run logs to the Airon tracker.
#   git clone -b kezek-r1 https://github.com/karl4th/ke-zek.git ~/ke-zek
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_packft.sh > /data/v3_packft.log 2>&1 &
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
from huggingface_hub import hf_hub_download as d, snapshot_download
t = os.environ["HF_TOKEN"]
d("manifestro/kezek-k-r1", "k-v3-v3s1c/best.pt", local_dir="/data/init", token=t)
d("manifestro/kezek-k-r1", "k-v3-ecapa1/ecapa.pt", local_dir="/data/init", token=t)
snapshot_download("manifestro/kezek-v1-bench", repo_type="dataset", revision="d0fdd6d90d460430724788198518677793047d64",
                  local_dir="/data/bench", token=t)
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
mkdir -p /data/fe
export KEZEK_PACK_SOLO=1 KEZEK_ECAPA=/data/init/k-v3-ecapa1/ecapa.pt
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --conversations 113 --model $model --cache /data/fe/cache_packft"
$e --out /data/fe/packft_clean.json > /data/fe/packft_clean.log 2>&1 &
$e --aug-root /data/aug --out /data/fe/packft_rec.json > /data/fe/packft_rec.log 2>&1 &
bash "$here/s1_bench.sh" /data/fe/bench_packft.json packft=$model > /data/fe/bench_packft.log 2>&1 &
wait
for r in "" "--rec"; do
  echo "packft $r $(python experiments/K-M1/s1_swap_diag.py --data /data/oy16 --split-file $split --cache /data/fe/cache_packft $r 2>/dev/null | tail -1)"
done >> /data/fe/diag.txt
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("/data/fe/packft_clean.json", "/data/fe/packft_rec.json", "/data/fe/bench_packft.json"):
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-ecapa1/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
api.upload_file(path_or_fileobj="/data/fe/diag.txt", path_in_repo="k-v3-ecapa1/diag_packft.txt", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 PACKFT DONE"
