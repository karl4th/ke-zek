#!/usr/bin/env bash
# Convert the first PARTS Open Yap parts listed in SHA256SUMS (default: all), JOBS at a time,
# skipping finished parts. About 2.2 GB of 16 kHz arrays per part.
#   bash experiments/K-R1/stage.sh s3://bucket/datasets/open-yap-... /data/oy16 [JOBS] [PARTS]
set -euo pipefail
src=${1%/}
out=$2
jobs=${3:-3}
parts=${4:-1000}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/.parts_done" "$out/.logs"
aws s3 cp "$src/SHA256SUMS" "$out/SHA256SUMS" --only-show-errors

one() {
  local sum=$1 file=$2
  [[ -e "$out/.parts_done/$file" ]] && return 0
  aws s3 cp "$src/$file" - --only-show-errors \
    | python "$here/convert.py" --out "$out" --part "$file" --sha256 "$sum" \
    > "$out/.logs/$file.log" 2>&1 || { echo "FAILED $file"; return 1; }
  echo "done $file"
}
export -f one
export src out here
grep ' part-.*\.tar$' "$out/SHA256SUMS" | head -n "$parts" | xargs -P "$jobs" -n 2 bash -c 'one "$0" "$1"'
echo "parts done: $(ls "$out/.parts_done" | wc -l)"
