#!/usr/bin/env bash
# Download, verify, and extract archive parts one at a time on a training machine.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 S3_ARCHIVE_PREFIX OUTPUT_DIR PART_COUNT" >&2
  exit 2
fi

source_uri=${1%/}
output_dir=$2
part_count=$3
if ! [[ $part_count =~ ^[1-9][0-9]*$ ]]; then
  echo "PART_COUNT must be a positive integer" >&2
  exit 2
fi

mkdir -p "$output_dir/.parts_done"
output_dir=$(cd "$output_dir" && pwd)
source_file="$output_dir/.source-uri"
if [[ -e $source_file && $(cat "$source_file") != "$source_uri" ]]; then
  echo "Output directory belongs to another source URI" >&2
  exit 2
fi
printf '%s\n' "$source_uri" > "$source_file"
aws s3 cp "$source_uri/SHA256SUMS" "$output_dir/SHA256SUMS" --only-show-errors

processed=0
while read -r expected file; do
  [[ $file == part-*.tar ]] || continue
  ((processed += 1))
  ((processed <= part_count)) || break
  [[ -e "$output_dir/.parts_done/$file" ]] && continue

  echo "Preparing $file ($processed/$part_count)"
  aws s3 cp "$source_uri/$file" "$output_dir/$file" --only-show-errors
  (cd "$output_dir" && printf '%s  %s\n' "$expected" "$file" | sha256sum -c -)
  tar -tf "$output_dir/$file" | awk '
    $0 ~ /^\// || $0 ~ /(^|\/)\.\.(\/|$)/ { bad = 1 }
    END { exit bad }
  '
  tar -xf "$output_dir/$file" -C "$output_dir"
  rm -- "$output_dir/$file"
  touch "$output_dir/.parts_done/$file"
done < "$output_dir/SHA256SUMS"

if ((processed < part_count)); then
  echo "Only $processed parts are listed in SHA256SUMS" >&2
  exit 1
fi
echo "Prepared $part_count archive parts in $output_dir"
