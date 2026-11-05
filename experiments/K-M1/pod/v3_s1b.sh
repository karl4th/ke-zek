#!/usr/bin/env bash
# Kezek v3 stage 1, run b: the v3s1 recipe plus monologue spans (25% of spans with one track
# silenced; Open Yap has ~4% single-speaker 16 s spans, podcasts many), started from the latest
# v3s1 weights. Then full held-out conversations, kezek-v1-bench, checkpoints to HF k-v3-s1b/.
#   nohup bash experiments/K-M1/pod/v3_s1b.sh > /data/v3_s1b.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../.."
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-S1 stage 1: profiles and activity (WavLM + Conformer, 16 s)"
split=experiments/K-R1/split_round1.json
runs=/data/runs
name=${RUN:-v3s1b}
init=${INIT:-$runs/v3s1/model.pt}
steps=${STEPS:-18000}
mkdir -p $runs/$name
[[ -e $runs/$name/init.pt ]] || cp "$init" $runs/$name/init.pt
python experiments/K-M1/train_m2s.py --data /data/oy16 --split-file $split --stage enc --features wavlm \
  --backbone conformer --slot-embedding ecapa --solo-threshold 0.5 --span-s 16 --bc-weight 5 \
  --single-p 0.25 ${EXTRA:-} --amp --workers 3 --experiment-tags kezek-v3,stage1,monologue \
  --aug-root /data/aug --steps $steps --bs 8 --lr-enc 2e-4 --eval-every 1000 \
  --weights $runs/v3s1_calib/calibration.json --init-model $runs/$name/init.pt \
  --resume --out $runs/$name >> $runs/$name/log.txt 2>&1
grep -q "TRAINING DONE" $runs/$name/log.txt || { echo "training failed"; exit 1; }
e=experiments/K-M1/s1_eval_full.py
python $e --data /data/oy16 --split-file $split --model $runs/$name/best.pt --conversations 113 \
  --step $steps --out $runs/$name/full_val.json > $runs/$name/full_val.log 2>&1
python $e --data /data/oy16 --split-file $split --model $runs/$name/best.pt --conversations 113 \
  --aug-root /data/aug --step $steps --out $runs/$name/full_val_rec.json > $runs/$name/full_val_rec.log 2>&1
bash experiments/K-M1/pod/s1_bench.sh $runs/$name/bench.json $name=$runs/$name/best.pt > $runs/$name/bench.log 2>&1
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("best.pt", "model.pt", "eval_log.json", "manifest.json", "full_val.json", "full_val_rec.json", "bench.json"):
    p = f"$runs/$name/{f}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"k-v3-$name/{f}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "$name DONE"
