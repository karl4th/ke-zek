#!/usr/bin/env bash
# Two-block Kezek on Open Yap rendered as mono recordings (K-R2 render), all training data:
# block 1 from the WavLM encoder, separator from a K-R2 checkpoint; each stage calibrates its
# weights first (256 batches), then trains with them frozen.
#   bash experiments/K-M1/run_m2s_render.sh DATA SPLIT AUG RUNS BLOCK1_MODEL K_R2_TSE
set -uo pipefail
data=$1
split=$2
aug=$3
runs=$4
block1=$5
tse=$6
here=$(cd "$(dirname "$0")" && pwd)
export AIRON_EXPERIMENT="K-M1 two-block Kezek on Open Yap"
common=(--data "$data" --split-file "$split" --aug-root "$aug" --features wavlm --slot-embedding ecapa
        --solo-threshold 0.5 --amp --workers 3)

stage() {  # name, calibration batch, extra
  local name=$1 cal_bs=$2
  shift 2
  grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null && return 0
  if [[ ! -e $runs/${name}_calib/calibration.json ]]; then
    AIRON_API_KEY= python "$here/train_m2s.py" "${common[@]}" --out "$runs/${name}_calib" \
      --calibrate 256 "$@" --bs "$cal_bs" > "$runs/${name}_calib.log" 2>&1
  fi
  mkdir -p "$runs/$name"
  python "$here/train_m2s.py" "${common[@]}" --out "$runs/$name" --resume --eval-every 1000 \
    --weights "$runs/${name}_calib/calibration.json" "$@" >> "$runs/$name/log.txt" 2>&1
}

stage m2s_r_enc 8 --stage enc --steps 2000 --bs 8 --lr-enc 1e-4 --init-model "$block1"
stage m2s_r_sep 2 --stage sep --steps 5000 --bs 4 --chunks 2 --lr-sep 1e-4 \
  --init-model "$runs/m2s_r_enc/model.pt" --init-tse "$tse"
stage m2s_r_e2e 2 --stage e2e --steps 5000 --bs 4 --chunks 2 --lr-enc 5e-5 --lr-sep 1e-4 \
  --init-model "$runs/m2s_r_sep/model.pt"
echo "M2S RENDER DONE"
