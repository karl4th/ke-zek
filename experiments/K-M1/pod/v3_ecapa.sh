#!/usr/bin/env bash
# Kezek v3 step 1b: voice embeddings of stage 1 on a fresh Airon AWS instance, end to end.
#   A. ECAPA on packed solo audio (no zero gaps) with the v3s1c model: full held-out
#      conversations, clean and rendered, link diagnosis.
#   B. ECAPA fine-tuned on VoxCeleb2 pieces under the recording render (ecapa_ft.py).
#   C. A again with the fine-tuned ECAPA; kezek-v1-bench; ecapa.pt to HF k-v3-ecapa1/.
# Every run logs to the Airon tracker.
#   git clone -b kezek-r1 https://github.com/karl4th/ke-zek.git ~/ke-zek
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_ecapa.sh > /data/v3_ecapa.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-S1 stage 1: profiles and activity (WavLM + Conformer, 16 s)"
pip install -q pyarrow
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import hf_hub_download as d, snapshot_download
t = os.environ["HF_TOKEN"]
d("manifestro/kezek-k-r1", "k-v3-v3s1c/best.pt", local_dir="/data/init", token=t)
snapshot_download("manifestro/kezek-v1-bench", repo_type="dataset", revision="d0fdd6d90d460430724788198518677793047d64",
                  local_dir="/data/bench", token=t)
print("MODELS READY")
PY
split=$repo/experiments/K-R1/split_round1.json
# only validation and test conversations are needed here: all of Open Yap does not fit next to VoxCeleb2
python -c "import json; d = json.load(open('$split')); print('\n'.join('conv_' + c for c in d['val'] + d['test']))" > /data/keep.txt
export KEEP=/data/keep.txt
[ -d /data/oy16 ] && (cd /data/oy16 && ls -d conv_* | grep -vxFf "$KEEP" | xargs -r rm -rf)
grep -q "AUG READY" /data/get_aug.log 2>/dev/null || ( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
grep -q "PREP DONE" /data/vox_prep.log 2>/dev/null || ( python experiments/K-M1/ecapa_ft.py --prep --vox /data/vox --shards 10 > /data/vox_prep.log 2>&1 ) &
for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 6 >> /data/get_data.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
grep -q "PREP DONE" /data/vox_prep.log || { echo "voxceleb prep failed"; exit 1; }
model=/data/init/k-v3-v3s1c/best.pt
mkdir -p /data/fe /data/runs

full() {  # name: full eval clean + rendered with a window cache, then link diagnosis
  local n=$1 e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --conversations 113 --model $model --cache /data/fe/cache_$1"
  $e --out /data/fe/${n}_clean.json > /data/fe/${n}_clean.log 2>&1 &
  $e --aug-root /data/aug --out /data/fe/${n}_rec.json > /data/fe/${n}_rec.log 2>&1 &
  wait
  for r in "" "--rec"; do
    echo "$n $r $(python experiments/K-M1/s1_swap_diag.py --data /data/oy16 --split-file $split --cache /data/fe/cache_$n $r 2>/dev/null | tail -1)"
  done >> /data/fe/diag.txt
}

export KEZEK_PACK_SOLO=1
full pack
python experiments/K-M1/ecapa_ft.py --vox /data/vox --aug-root /data/aug --out /data/runs/ecapa1 \
  > /data/runs/ecapa1.log 2>&1
grep -q "TRAINING DONE" /data/runs/ecapa1.log || { echo "ecapa training failed"; exit 1; }
export KEZEK_ECAPA=/data/runs/ecapa1/ecapa.pt
full packft
bash "$here/s1_bench.sh" /data/fe/bench_packft.json \
  packft=$model > /data/fe/bench_packft.log 2>&1
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("/data/runs/ecapa1/ecapa.pt", "/data/runs/ecapa1.log", "/data/fe/diag.txt", "/data/fe/pack_clean.json",
          "/data/fe/pack_rec.json", "/data/fe/packft_clean.json", "/data/fe/packft_rec.json", "/data/fe/bench_packft.json"):
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-ecapa1/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 ECAPA DONE"
