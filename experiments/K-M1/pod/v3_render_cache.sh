#!/usr/bin/env bash
# Render the 113 validation recordings once (render_cache.py, three at a time) and put them on HF
# (dataset manifestro/kezek-render-cache, render_all/), so rendered evaluations read them instead
# of rendering again (s1_eval_full.py --x-cache).
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_render_cache.sh > /data/v3_render_cache.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export KEZEK_LABELS=words KEZEK_WORD_GAP=0.3
split=$repo/experiments/K-R1/split_round1.json
python -c "import json; d = json.load(open('$split')); print('\n'.join('conv_' + c for c in d['val'] + d['test']))" > /data/keep.txt
export KEEP=/data/keep.txt
grep -q "AUG READY" /data/get_aug.log 2>/dev/null || ( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 6 >> /data/get_data.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
out=/data/xcache/render_all
for s in 0 1 2; do
  python experiments/K-M1/render_cache.py --data /data/oy16 --split-file $split --aug-root /data/aug \
    --out $out --shard $s --shards 3 > /data/render_$s.log 2>&1 &
done
wait
[ "$(ls $out/*_all.npy | wc -l)" -ge 113 ] || { echo "render incomplete"; exit 1; }
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("manifestro/kezek-render-cache", repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="$out", path_in_repo="render_all", repo_id="manifestro/kezek-render-cache", repo_type="dataset")
print("UPLOADED")
PY
echo "RENDER CACHE DONE"
