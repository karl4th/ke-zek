#!/usr/bin/env bash
# Long-form comparison of every K-R1 method on one split: 60 s excerpts, 20 s enrollment.
#   bash experiments/K-R1/final_eval.sh /data/oy16 /data/init /data/runs /data/final val
#   bash experiments/K-R1/final_eval.sh /data/oy16 /data/init /data/runs /data/final test SEP TSE
# For test (read once, with --final) pass the separator and extractor chosen on validation.
set -uo pipefail
data=$1
init=$2
runs=$3
out=$4
split=$5
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/$split"
final=()
[[ $split == test ]] && final=(--final)
sep=${6:-$runs/s3_sep_kt0c_12s/sep.pt}
[[ -e $sep ]] || sep=$runs/s1_sep_kt0c/sep.pt
tse=${7:-$runs/r1_kt0c_state/kezek_tse.pt}

run() {  # name, arguments
  local name=$1
  shift
  [[ -e $out/$split/$name.json ]] && return 0
  python "$here/evaluate_long.py" --data "$data" --split "$split" "${final[@]}" \
    --out "$out/$split/$name.json" "$@" > "$out/$split/$name.log" 2>&1
}

run mixture --method mixture
run kt0c --method mono --model "$init/kezek-k-t0c/kezek_tse.pt"
run kt1 --method mono --model "$init/kezek-k-t1/kezek_tse.pt"
run r1 --method mono --model "$tse"
[[ -e $runs/r5_kt0c_state_dual/kezek_tse.pt ]] && run r5_dual --method mono \
  --model "$runs/r5_kt0c_state_dual/kezek_tse.pt"
run block --method block --model "$sep"
run block_cross15 --method block --model "$sep" --cross-db 15
run block_voice --method block --model "$sep" --no-conversation
run hybrid --method hybrid --model "$sep" --tse-model "$tse"
run hybrid_cross15 --method hybrid --model "$sep" --tse-model "$tse" --cross-db 15
run hybrid_voice --method hybrid --model "$sep" --tse-model "$tse" --no-conversation
run oracle_block --method oracle_block
run oracle_block_voice --method oracle_block --no-conversation
echo "FINAL EVAL DONE $split"
