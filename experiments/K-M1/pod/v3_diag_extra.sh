#!/usr/bin/env bash
# Kezek v3 stage 1 diagnosis (K-V3-S1-DIAG), rendered cells and extra questions: the v3s1c model
# with packed solo audio, no training, full recordings of the validation conversations.
#   rendered (recordings over 30 min rendered in 20 min pieces): base (solo threshold 0.5),
#   labelled activity (oracle), solo threshold 0.9, only the per-voice part of the chain, only the
#   mixture part;
#   clean: F1 with a 200 ms collar and where false positives fall, oracle embeddings with
#   predicted activity, solo thresholds 0.3 and 0.2, the train/val gap (30 train conversations).
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
v="$e --conversations 113"
rendered() {  # one at a time: a rendered piece takes several GB of RAM
  c="--cache /data/dx/cache_h8 --aug-root /data/aug"
  $v $c --tag _pred05 --out /data/dx/pred05_rec.json > /data/dx/pred05_rec.log 2>&1
  $v $c --reembed label_full --tag _label --out /data/dx/label_rec.json > /data/dx/label_rec.log 2>&1
  $v $c --reembed pred --solo-thr 0.9 --tag _pred09 --out /data/dx/pred09_rec.json > /data/dx/pred09_rec.log 2>&1
  for part in sources finish; do
    $v --aug-root /data/aug --render-part $part --tag _rec_$part --out /data/dx/rec_${part}.json > /data/dx/rec_${part}.log 2>&1
  done
}
clean() {
  c="--cache /data/dx/cache_h8"
  $v $c --tag _collar --out /data/dx/collar_clean.json > /data/dx/collar_clean.log 2>&1
  $v $c --reembed label --tag _oracle_emb --out /data/dx/oracle_emb_clean.json > /data/dx/oracle_emb_clean.log 2>&1
  for t in 0.3 0.2; do
    $v $c --reembed pred --solo-thr $t --tag _solo$t --out /data/dx/solo${t}_clean.json > /data/dx/solo${t}_clean.log 2>&1
  done
  $e --split train --conversations 30 --tag _train30 --out /data/dx/train30_clean.json > /data/dx/train30_clean.log 2>&1
}
rendered &
clean &
wait
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
