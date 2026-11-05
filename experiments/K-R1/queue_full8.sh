#!/usr/bin/env bash
# The full R5 recipe with an 8 s training window, alongside queue_full.sh's 4 s run: same start
# (K-T0c), steps, data, validation and test; batch 4 x 8 s holds as much audio as 8 x 4 s.
#   bash experiments/K-R1/queue_full8.sh /data/oy16 /data/init /data/runs /data/final SPLIT_FILE
set -uo pipefail
data=$1
init=$2
runs=$3
out=$4
split=$5
here=$(cd "$(dirname "$0")" && pwd)
name=r5_full_8s_k0

if ! grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null; then
  mkdir -p "$runs/$name"
  python "$here/train.py" --data "$data" --out "$runs/$name" --resume --split-file "$split" \
    --steps 40000 --init "$init/kezek-k-t0c/kezek_tse.pt" --flip-init --lr 1e-4 --dual 0.3 \
    --crop-s 8 --bs 4 --amp >> "$runs/$name/log.txt" 2>&1
fi
for s in val test; do
  mkdir -p "$out/$s"
  [[ -e $out/$s/$name.json ]] && continue
  final=()
  [[ $s == test ]] && final=(--final)
  python "$here/evaluate_long.py" --data "$data" --split "$s" "${final[@]}" --split-file "$split" \
    --method mono --model "$runs/$name/kezek_tse.pt" --mono-win-s 8 \
    --out "$out/$s/$name.json" > "$out/$s/$name.log" 2>&1
done
echo "FULL 8S DONE"
