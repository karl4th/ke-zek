#!/usr/bin/env bash
# Stage 1 checkpoint(s) on kezek-v1-bench next to the K-M1 block 1 baseline (same script).
#   bash experiments/K-M1/pod/s1_bench.sh OUT name=model.pt [name=model.pt ...]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../.."
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-S1 stage 1: profiles and activity (WavLM + Conformer, 16 s)"
out=$1
shift
python experiments/K-M1/bench_eval.py --bench /data/bench --revision d0fdd6d90d460430724788198518677793047d64 \
  --out "$out" --s1 "$@"
