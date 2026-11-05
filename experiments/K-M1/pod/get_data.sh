#!/usr/bin/env bash
# Open Yap 1K in training format (16 kHz npy + 10 ms activity, packed by part) from S3, checked
# against SHA256SUMS, unpacked into OUT (default /data/oy16), JOBS parts at a time. When the
# instance role cannot read the converted copy, the source delivery is converted on the instance
# instead (experiments/K-R1/stage.sh, about an hour).
#   bash experiments/K-M1/pod/get_data.sh [OUT] [JOBS]
set -euo pipefail
bucket=s3://manifestro-dataset-876108973178-us-east-1-an/datasets
src=$bucket/open-yap-1k_2026-11-04_wav_0e7bcf1e_0d3abb77_16k-npy-act
out=${1:-/data/oy16}
jobs=${2:-6}
mkdir -p "$out/.parts_done" /data/pack
if ! aws s3 cp "$src/SHA256SUMS" "$out/SHA256SUMS" --only-show-errors 2>/dev/null; then
  echo "converted copy not readable: converting the source delivery"
  here=$(cd "$(dirname "$0")" && pwd)
  for i in 1 2 3 4; do
    bash "$here/../../K-R1/stage.sh" "$bucket/open-yap-1k_2026-11-04_wav_0e7bcf1e_0d3abb77" "$out" 2 116 || true
    rm -rf "$out"/.tmp_conv_*
    [ "$(ls "$out/.parts_done" | wc -l)" -ge 115 ] && break
  done
  echo "conversations: $(ls -d "$out"/conv_* | wc -l)"
  exit 0
fi
one() {  # stream: S3 -> SHA-256 on the fly -> tar; a mismatch removes what the part wrote
  local sum=$1 file=$2
  [[ -e "$out/.parts_done/$file" ]] && return 0
  aws s3 cp "$src/$file" - --only-show-errors \
    | python3 -c 'import hashlib, sys
h = hashlib.sha256()
while b := sys.stdin.buffer.read(1 << 22):
    h.update(b)
    sys.stdout.buffer.write(b)
open(sys.argv[1], "w").write(h.hexdigest())' "/data/pack/$file.sha" \
    | tar xv -C "$out" > "/data/pack/$file.list"
  if [[ "$(cat "/data/pack/$file.sha" 2>/dev/null)" != "$sum" ]]; then
    echo "SHA MISMATCH $file"
    (cd "$out" && cut -d/ -f1 "/data/pack/$file.list" | sort -u | xargs -r rm -rf)
    return 1
  fi
  touch "$out/.parts_done/$file" && rm -f "/data/pack/$file.sha" "/data/pack/$file.list"
  echo "done $file"
}
export -f one
export src out
grep ' part-.*\.tar$' "$out/SHA256SUMS" | xargs -P "$jobs" -n 2 bash -c 'one "$0" "$1"'
echo "parts done: $(ls "$out/.parts_done" | wc -l) of $(grep -c ' part-.*\.tar$' "$out/SHA256SUMS")"
echo "conversations: $(ls -d "$out"/conv_* | wc -l)"
