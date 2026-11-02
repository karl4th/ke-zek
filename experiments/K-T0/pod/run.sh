#!/bin/bash
# K-T0c on a pod: 500k steps on train-clean-100 + 360, continues from last.pt, restarts after a crash (up to 20 times).
# OMP_NUM_THREADS=1 matters: a pod shows 192 cores but has a quota of about 20 (0.33 -> 0.15 s/step on an A100).
. /workspace/ke-zek/experiments/K-T0/pod/env.sh
cd /root/d || exit 1
export AIRON_URL=https://api.manifestro.ci OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=kezek-run-360
for try in $(seq 1 20); do
  echo "=== start try $try $(date -Is)" >> $O/log.txt
  python /workspace/ke-zek/experiments/K-T0/train_scale.py \
    --train-roots extracted/LibriSpeech/train-clean-100 extracted/LibriSpeech/train-clean-360 --test-root extracted/LibriSpeech/dev-clean \
    --steps 500000 --bs 8 --accum 1 --eval-every 5000 --first-eval 500 --seed 0 --workers 16 --fast \
    --out $O --run-name k-t0c-500k-a100-git --resume >> $O/log.txt 2>&1 && break
  echo "=== crashed, exit $? $(date -Is)" >> $O/log.txt
  sleep 30
done
