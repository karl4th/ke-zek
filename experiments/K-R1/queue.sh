#!/usr/bin/env bash
# The K-R1 comparison, one run after another on one GPU. Finished runs (TRAINING DONE in their
# log) are skipped, interrupted ones resume from last.pt.
#   bash experiments/K-R1/queue.sh /data/oy16 /data/init /data/runs [STEPS]
set -uo pipefail
data=$1
init=$2
runs=$3
steps=${4:-30000}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$runs"

evaluate() {  # name, model, [arch]
  local out="$runs/zero_$1.json"
  [[ -e $out ]] && return 0
  python "$here/evaluate_r1.py" --data "$data" --model "$2" --arch "${3:-kezek}" --out "$out" \
    > "$runs/zero_$1.log" 2>&1
}

train() {  # name, extra arguments
  local name=$1
  shift
  grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null && return 0
  mkdir -p "$runs/$name"
  python "$here/train.py" --data "$data" --out "$runs/$name" --steps "$steps" --resume "$@" \
    >> "$runs/$name/log.txt" 2>&1
}

evaluate mixture mixture
evaluate kt0c "$init/kezek-k-t0c/kezek_tse.pt"
evaluate kt1 "$init/kezek-k-t1/kezek_tse.pt"

kt0c=(--init "$init/kezek-k-t0c/kezek_tse.pt" --flip-init)
train r1_kt0c_state "${kt0c[@]}" --lr 1e-4 --amp
train r5_kt0c_state_dual "${kt0c[@]}" --lr 1e-4 --dual 0.3 --amp
train r2_scratch_state --lr 1e-3 --warmup 2000 --amp
train r3_band_state --arch band --lr 5e-4 --warmup 2000 --amp
train r4_kt1_state --init "$init/kezek-k-t1/kezek_tse.pt" --lr 1e-4 --amp
train r6_kt0c_kt1loss "${kt0c[@]}" --lr 1e-4 --loss kt1 --stop-at 5000 --amp
echo "QUEUE DONE"
