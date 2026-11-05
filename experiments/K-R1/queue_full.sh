#!/usr/bin/env bash
# The selected K-R1 recipe (R5) trained in full: all Open Yap parts, validation and test kept as
# in SPLIT_FILE so results compare with the first round; then the same with an 8 s window,
# started from the 4 s model; then both scored on validation and once on test.
#   bash experiments/K-R1/queue_full.sh /data/oy16 /data/init /data/runs /data/final SPLIT_FILE
set -uo pipefail
data=$1
init=$2
runs=$3
out=$4
split=$5
here=$(cd "$(dirname "$0")" && pwd)

train() {  # name, extra arguments
  local name=$1
  shift
  grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null && return 0
  mkdir -p "$runs/$name"
  python "$here/train.py" --data "$data" --out "$runs/$name" --resume --split-file "$split" "$@" \
    >> "$runs/$name/log.txt" 2>&1
}

evaluate() {  # split, name, arguments
  local s=$1 name=$2
  shift 2
  mkdir -p "$out/$s"
  [[ -e $out/$s/$name.json ]] && return 0
  local final=()
  [[ $s == test ]] && final=(--final)
  python "$here/evaluate_long.py" --data "$data" --split "$s" "${final[@]}" --split-file "$split" \
    --out "$out/$s/$name.json" "$@" > "$out/$s/$name.log" 2>&1
}

train r5_full --steps 40000 --init "$init/kezek-k-t0c/kezek_tse.pt" --flip-init --lr 1e-4 \
  --dual 0.3 --amp
train r5_full_8s --steps 15000 --init "$runs/r5_full/kezek_tse.pt" --crop-s 8 --bs 4 --lr 5e-5 \
  --warmup 500 --dual 0.3 --amp
for s in val test; do
  evaluate "$s" r5_full --method mono --model "$runs/r5_full/kezek_tse.pt"
  evaluate "$s" r5_full_8s --method mono --model "$runs/r5_full_8s/kezek_tse.pt" --mono-win-s 8
  evaluate "$s" mixture --method mixture
done
echo "FULL DONE"
