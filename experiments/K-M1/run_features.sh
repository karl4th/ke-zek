#!/usr/bin/env bash
# Block 1 input comparison: the same activity network and schedule on mel spectrogram, frozen
# WavLM Base+ and frozen Mimi latents; reported by activity F1 and same-span identity accuracy
# of the slot embeddings (ECAPA on confidently solo frames).
#   bash experiments/K-M1/run_features.sh DATA SPLIT RUNS [STEPS]
set -uo pipefail
data=$1
split=$2
runs=$3
steps=${4:-2000}
here=$(cd "$(dirname "$0")" && pwd)
export AIRON_EXPERIMENT="K-M1 two-block Kezek on Open Yap"
printf '{"weights": {"act": 1.0, "id": 0.62}}\n' > "$runs/features_weights.json"
for f in mel wavlm mimi; do
  name=m2s_feat_${f}_s${steps}
  grep -q "TRAINING DONE" "$runs/$name/log.txt" 2>/dev/null && continue
  mkdir -p "$runs/$name"
  python "$here/train_m2s.py" --stage enc --features "$f" --data "$data" --split-file "$split" \
    --train-hours 50 --slot-embedding ecapa --solo-threshold 0.5 --weights "$runs/features_weights.json" \
    --steps "$steps" --bs 8 --lr-enc 3e-4 --eval-every 500 --workers 3 --amp --resume \
    --out "$runs/$name" >> "$runs/$name/log.txt" 2>&1
done
echo "FEATURES DONE"
