#!/usr/bin/env bash
# Kezek v3 stage 1 (profiles and activity) on a fresh Airon AWS instance, end to end:
# setup, data, E0 overfit check, weight calibration, training, full-recording evaluation,
# checkpoints to HF (manifestro/kezek-k-r1, k-v3-s1/). Every run logs to the Airon tracker.
#   git clone -b kezek-r1 https://github.com/karl4th/ke-zek.git ~/ke-zek
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_s1.sh > /data/v3_s1.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-S1 stage 1: profiles and activity (WavLM + Conformer, 16 s)"
( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 8 >> /data/get_data.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
split=$repo/experiments/K-R1/split_round1.json
runs=/data/runs
mkdir -p $runs
common=(--data /data/oy16 --split-file "$split" --stage enc --features wavlm --backbone conformer
        --slot-embedding ecapa --solo-threshold 0.5 --span-s 16 --bc-weight 5 --amp --workers 3
        --experiment-tags kezek-v3,stage1)
t=experiments/K-M1/train_m2s.py

# E0: the network must fit 8 fixed windows (activity and short replies) before anything else
if ! grep -q "TRAINING DONE" $runs/v3s1_e0/log.txt 2>/dev/null; then
  mkdir -p $runs/v3s1_e0
  python $t "${common[@]}" --overfit 8 --steps 300 --bs 8 --lr-enc 1e-3 \
    --out $runs/v3s1_e0 > $runs/v3s1_e0/log.txt 2>&1
fi
grep "EVAL step 300" $runs/v3s1_e0/log.txt || { echo "E0 failed"; exit 1; }

# weight calibration (activity vs identity), then training with the weights frozen
if [[ ! -e $runs/v3s1_calib/calibration.json ]]; then
  AIRON_API_KEY= python $t "${common[@]}" --aug-root /data/aug --calibrate 256 --steps 20000 --bs 8 \
    --out $runs/v3s1_calib > $runs/v3s1_calib.log 2>&1
fi
[[ -e $runs/v3s1_calib/calibration.json ]] || { echo "calibration failed"; exit 1; }
mkdir -p $runs/v3s1
python $t "${common[@]}" --aug-root /data/aug --steps 20000 --bs 8 --lr-enc 3e-4 --eval-every 1000 \
  --weights $runs/v3s1_calib/calibration.json --resume --out $runs/v3s1 >> $runs/v3s1/log.txt 2>&1
grep -q "TRAINING DONE" $runs/v3s1/log.txt || { echo "training failed"; exit 1; }

# whole held-out conversations with the recording memory, clean and rendered
e=experiments/K-M1/s1_eval_full.py
python $e --data /data/oy16 --split-file "$split" --model $runs/v3s1/best.pt --conversations 113 \
  --step 20000 --out $runs/v3s1/full_val.json > $runs/v3s1/full_val.log 2>&1
python $e --data /data/oy16 --split-file "$split" --model $runs/v3s1/best.pt --conversations 113 \
  --aug-root /data/aug --step 20000 --out $runs/v3s1/full_val_rec.json > $runs/v3s1/full_val_rec.log 2>&1

python - <<'PY'
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("best.pt", "model.pt", "eval_log.json", "manifest.json", "full_val.json", "full_val_rec.json"):
    p = f"/data/runs/v3s1/{f}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"k-v3-s1/{f}", repo_id="manifestro/kezek-k-r1")
for f in ("calibration.json",):
    api.upload_file(path_or_fileobj=f"/data/runs/v3s1_calib/{f}", path_in_repo=f"k-v3-s1/{f}",
                    repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "V3 S1 DONE"
