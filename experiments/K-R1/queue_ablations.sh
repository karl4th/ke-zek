#!/usr/bin/env bash
# The K-R1 ablations, for a second instance while the first runs queue2.sh's main line.
# Same names and settings as in queue2.sh, so either instance's results can stand in.
#   bash experiments/K-R1/queue_ablations.sh /data/oy16 /data/init /data/runs [STEPS] [ABLATION_STEPS]
set -uo pipefail
data=$1
init=$2
runs=$3
steps=${4:-30000}
short=${5:-15000}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$runs"

train() {  # script, name, steps, extra arguments
  local script=$1 name=$2 n=$3
  shift 3
  grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null && return 0
  mkdir -p "$runs/$name"
  python "$here/$script" --data "$data" --out "$runs/$name" --steps "$n" --resume "$@" \
    >> "$runs/$name/log.txt" 2>&1
}

kt0c=(--init "$init/kezek-k-t0c/kezek_tse.pt" --flip-init)
train train.py r4_kt1_state "$short" --init "$init/kezek-k-t1/kezek_tse.pt" --lr 1e-4 --amp
train train.py r2_scratch_state "$short" --lr 1e-3 --warmup 2000 --amp
train train.py r3_band_state "$short" --arch band --lr 5e-4 --warmup 2000 --amp
train train.py r6_kt0c_kt1loss "$short" "${kt0c[@]}" --lr 1e-4 --loss kt1 --stop-at 5000 --amp
train train_sep.py s2_sep_band "$steps" --arch band --lr 5e-4 --warmup 2000 --amp
echo "ABLATIONS DONE"
