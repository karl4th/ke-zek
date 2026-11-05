#!/usr/bin/env bash
# Staged training of the two-block Kezek (M2S v0.1): E1 pilot (5 h) or E2 (50 h). Each stage first calibrates its objective weights
# on 256 batches from its own starting point (equal gradient scale), then trains with them frozen.
#   bash experiments/K-M1/run_stages.sh DATA SPLIT INIT RUNS HOURS ENC_STEPS SEP_STEPS E2E_STEPS
set -uo pipefail
data=$1
split=$2
init=$3
runs=$4
hours=${5:-5}
enc_steps=${6:-3000}
sep_steps=${7:-2000}
e2e_steps=${8:-3000}
here=$(cd "$(dirname "$0")" && pwd)
export AIRON_EXPERIMENT="K-M1 two-block Kezek on Open Yap"
tag=h${hours}
common=(--data "$data" --split-file "$split" --train-hours "$hours" --amp --workers 3
        --slot-embedding "${SLOTS:-ecapa}" --solo-threshold "${SOLO:-0.5}")

stage() {  # name, batch for calibration, extra (shared by calibration and training)
  local name=$1 cal_bs=$2
  shift 2
  if ! grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null; then
    if [[ ! -e $runs/${name}_calib/calibration.json ]]; then
      AIRON_API_KEY= python "$here/train_m2s.py" "${common[@]}" --out "$runs/${name}_calib" \
        --calibrate 256 --steps 256 "$@" --bs "$cal_bs" > "$runs/${name}_calib.log" 2>&1
    fi
    mkdir -p "$runs/$name"
    python "$here/train_m2s.py" "${common[@]}" --out "$runs/$name" --resume --eval-every 500 \
      --weights "$runs/${name}_calib/calibration.json" "$@" >> "$runs/$name/log.txt" 2>&1
  fi
}

stage m2s_${tag}_enc 8 --stage enc --steps "$enc_steps" --bs 8 --lr-enc 3e-4
stage m2s_${tag}_sep 2 --stage sep --steps "$sep_steps" --bs 4 --chunks 2 --lr-sep 3e-4 \
  --init-model "$runs/m2s_${tag}_enc/model.pt" --init-tse "$init/kezek-k-t0c/kezek_tse.pt" --flip-init
stage m2s_${tag}_e2e 2 --stage e2e --steps "$e2e_steps" --bs 4 --chunks 2 --lr-enc 1e-4 --lr-sep 3e-4 \
  --init-model "$runs/m2s_${tag}_sep/model.pt"
echo "STAGES DONE $tag"
