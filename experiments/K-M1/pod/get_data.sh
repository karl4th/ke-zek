#!/usr/bin/env bash
# Open Yap 1K in training format (16 kHz npy + 10 ms activity, packed by part) from S3, checked
# against SHA256SUMS, unpacked into OUT (default /data/oy16), JOBS parts at a time.
#   bash experiments/K-M1/pod/get_data.sh [OUT] [JOBS]
set -euo pipefail
src=s3://manifestro-dataset-876108973178-us-east-1-an/datasets/open-yap-1k_2026-11-04_wav_0e7bcf1e_0d3abb77_16k-npy-act
out=${1:-/data/oy16}
jobs=${2:-8}
mkdir -p "$out/.parts_done" /data/pack
aws s3 cp "$src/SHA256SUMS" "$out/SHA256SUMS" --only-show-errors
one() {
  local sum=$1 file=$2
  [[ -e "$out/.parts_done/$file" ]] && return 0
  aws s3 cp "$src/$file" "/data/pack/$file" --only-show-errors
  echo "$sum  /data/pack/$file" | sha256sum -c --quiet - || { echo "SHA MISMATCH $file"; rm -f "/data/pack/$file"; return 1; }
  tar xf "/data/pack/$file" -C "$out" && rm -f "/data/pack/$file" && touch "$out/.parts_done/$file"
  echo "done $file"
}
export -f one
export src out
grep ' part-.*\.tar$' "$out/SHA256SUMS" | xargs -P "$jobs" -n 2 bash -c 'one "$0" "$1"'
echo "parts done: $(ls "$out/.parts_done" | wc -l) of $(grep -c ' part-.*\.tar$' "$out/SHA256SUMS")"
echo "conversations: $(ls -d "$out"/conv_* | wc -l)"
