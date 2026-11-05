#!/usr/bin/env bash
# E1 pilot of the two-block Kezek (M2S v0.1 stages) with frozen calibrated weights.
#   bash experiments/K-M1/run_e1.sh /data/oy16 /data/split_round1.json /data/init /data/runs CALIBRATION [HOURS]
set -uo pipefail
data=$1
split=$2
init=$3
runs=$4
weights=$5
hours=${6:-5}
here=$(cd "$(dirname "$0")" && pwd)
export AIRON_EXPERIMENT="K-M1 two-block Kezek on Open Yap"
tag=e1_h${hours}
common=(--data "$data" --split-file "$split" --train-hours "$hours" --weights "$weights" --amp
        --workers 3 --eval-every 500 --resume)

stage() {  # name, extra
  local name=$1
  shift
  grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null && return 0
  mkdir -p "$runs/$name"
  python "$here/train_m2s.py" "${common[@]}" --out "$runs/$name" "$@" >> "$runs/$name/log.txt" 2>&1
}

stage m2s_${tag}_enc --stage enc --steps 1500 --bs 8 --lr-enc 3e-4
stage m2s_${tag}_sep --stage sep --steps 1500 --bs 4 --chunks 2 --lr-sep 3e-4 \
  --init-model "$runs/m2s_${tag}_enc/model.pt" --init-tse "$init/kezek-k-t0c/kezek_tse.pt" --flip-init
stage m2s_${tag}_e2e --stage e2e --steps 3000 --bs 4 --chunks 2 --lr-enc 1e-4 --lr-sep 3e-4 \
  --init-model "$runs/m2s_${tag}_sep/model.pt"
echo "E1 DONE"
