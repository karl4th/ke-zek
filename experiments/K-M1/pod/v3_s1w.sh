#!/usr/bin/env bash
# Kezek v3 stage 1 on word-gated activity labels (KEZEK_LABELS=words): the v3s1c recipe (WavLM +
# Conformer, powerset head, 16 s spans, monologue spans, short-reply weight 5) continued from
# v3s1c on labels without the crosstalk that the energy VAD of each close-talk track marks as
# speech. Then full recordings of the validation conversations (word labels: clean, rendered;
# energy labels: clean, for comparison) and kezek-v1-bench; checkpoints to HF k-v3-$RUN/.
# Every run logs to the Airon tracker.
#   RUN=v3s1w STEPS=12000 nohup bash ~/ke-zek/experiments/K-M1/pod/v3_s1w.sh > /data/v3_s1w.log 2>&1 &
# Env: RUN, STEPS, LR, INIT (HF path of the starting best.pt), EXTRA (more train_m2s flags), WORD_PAD.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-S1W stage 1 on word-gated activity labels"
export KEZEK_LABELS=words KEZEK_WORD_PAD=${WORD_PAD:-0.15}
name=${RUN:-v3s1w}
steps=${STEPS:-12000}
init=${INIT:-k-v3-v3s1c/best.pt}
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import hf_hub_download as d
t = os.environ["HF_TOKEN"]
d("manifestro/kezek-k-r1", "$init", local_dir="/data/init", token=t)
import json
os.makedirs("/data/init/k-v3-s1", exist_ok=True)  # the v3s1 calibration (manifest of v3s1c): never uploaded
json.dump({"weights": {"sep": 1.0, "act": 1.0, "id": 0.1, "mix": 0.1}}, open("/data/init/k-v3-s1/calibration.json", "w"))
d("manifestro/kezek-k-r1", "k-v3-v3s1c/best.pt", local_dir="/data/init", token=t)
from huggingface_hub import snapshot_download
snapshot_download("manifestro/kezek-v1-bench", repo_type="dataset", revision="d0fdd6d90d460430724788198518677793047d64",
                  local_dir="/data/bench", token=t)
print("MODELS READY")
PY
# training needs every conversation: drop a validation-only copy (its part markers) first
if [ "$(ls -d /data/oy16/conv_* 2>/dev/null | wc -l)" -lt 1000 ]; then rm -rf /data/oy16/.parts_done; fi
unset KEEP
grep -q "AUG READY" /data/get_aug.log 2>/dev/null || ( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 6 >> /data/get_data_full.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
[ "$(ls -d /data/oy16/conv_* | wc -l)" -ge 1100 ] || { echo "data incomplete"; exit 1; }
split=$repo/experiments/K-R1/split_round1.json
runs=/data/runs
mkdir -p $runs/$name
python experiments/K-M1/train_m2s.py --data /data/oy16 --split-file $split --stage enc --features wavlm \
  --backbone conformer --slot-embedding ecapa --solo-threshold 0.5 --span-s 16 --bc-weight 5 \
  --single-p 0.25 --act-head powerset ${EXTRA:-} --amp --workers 3 --experiment-tags kezek-v3,stage1,word-labels \
  --aug-root /data/aug --steps $steps --bs 8 --lr-enc ${LR:-2e-4} --eval-every 1000 \
  --weights /data/init/k-v3-s1/calibration.json --init-model /data/init/$init \
  --resume --out $runs/$name >> $runs/$name/log.txt 2>&1
grep -q "TRAINING DONE" $runs/$name/log.txt || { echo "training failed"; exit 1; }
export KEZEK_PACK_SOLO=1
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --max-minutes 100000 --conversations 113 --model $runs/$name/best.pt"
$e --cache $runs/$name/cache --tag _words --out $runs/$name/full_val.json > $runs/$name/full_val.log 2>&1
KEZEK_LABELS=energy $e --cache $runs/$name/cache --tag _energylabels --out $runs/$name/full_val_energy.json > $runs/$name/full_val_energy.log 2>&1
$e --cache $runs/$name/cache --aug-root /data/aug --tag _words --out $runs/$name/full_val_rec.json > $runs/$name/full_val_rec.log 2>&1
bash "$here/s1_bench.sh" $runs/$name/bench.json $name=$runs/$name/best.pt > $runs/$name/bench.log 2>&1
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("best.pt", "model.pt", "eval_log.json", "full_val.json", "full_val_energy.json", "full_val_rec.json", "bench.json"):
    p = f"$runs/$name/{f}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"k-v3-$name/{f}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "$name DONE"
