#!/bin/bash
# K-T1 on a pod: 100k steps from the K-T0c checkpoint. Continues from last.pt after a crash (up to
# 20 times); a watchdog stop (exit 3) is a bug to fix, not a crash, so it is never restarted.
# OMP_NUM_THREADS=1: a pod shows many cores but has a quota of about 20.
. /workspace/ke-zek/experiments/K-T0/pod/env.sh
cd /root/d || exit 1
export AIRON_URL=https://api.manifestro.ci OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=${O:-kt1-run}
SEED=${SEED:-0}
mkdir -p $O
for try in $(seq 1 20); do
  echo "=== start try $try $(date -Is)" >> $O/log.txt
  python /workspace/ke-zek/experiments/K-T1/train.py --init /root/d/init/kezek_tse.pt \
    --train-roots extracted/LibriSpeech/train-clean-100 extracted/LibriSpeech/train-clean-360 \
    --val-root extracted/LibriSpeech/dev-clean \
    --rir-dirs extracted/RIRS_NOISES/simulated_rirs extracted/RIRS_NOISES/real_rirs_isotropic_noises \
    --noise-dirs extracted/musan/noise --music-dirs extracted/musan/music \
    --steps 100000 --bs 8 --eval-every 5000 --seed $SEED --workers 16 --fast \
    --out $O --run-name k-t1-seed$SEED --resume >> $O/log.txt 2>&1
  code=$?
  [ $code -eq 0 ] && break
  if [ $code -eq 3 ]; then echo "=== watchdog stop $(date -Is), not restarted" >> $O/log.txt; break; fi
  echo "=== crashed, exit $code $(date -Is)" >> $O/log.txt
  sleep 30
done
