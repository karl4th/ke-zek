#!/usr/bin/env bash
# Kezek v3 stage 1, second pass: ProfileHead (TS-VAD style) on a frozen stage 1 encoder trained on
# word-gated labels (train_tsvad.py), then full recordings of the validation conversations
# (clean, rendered) with the head after linking, and kezek-v1-bench of the encoder for reference.
# Checkpoints and results to HF k-v3-$RUN/. Every run logs to the Airon tracker.
#   RUN=tsvad1 ENC=k-v3-v3s1w_g03/best.pt nohup bash ~/ke-zek/experiments/K-M1/pod/v3_tsvad.sh > /data/v3_tsvad.log 2>&1 &
# Env: RUN, ENC (HF path of the encoder best.pt), STEPS, EXTRA (more train_tsvad flags), WORD_GAP,
#      LINK (link_windows JSON for the evaluation).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export AIRON_EXPERIMENT="K-V3-S1W stage 1 on word-gated activity labels"
export KEZEK_LABELS=words KEZEK_WORD_GAP=${WORD_GAP:-0.3}
name=${RUN:-tsvad1}
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import hf_hub_download as d
d("manifestro/kezek-k-r1", "$ENC", local_dir="/data/init", token=os.environ["HF_TOKEN"])
print("MODELS READY")
PY
[ "$(ls -d /data/oy16/conv_* 2>/dev/null | wc -l)" -ge 1100 ] || { echo "full Open Yap needed in /data/oy16"; exit 1; }
grep -q "AUG READY" /data/get_aug.log || { echo "aug needed in /data/aug"; exit 1; }
split=$repo/experiments/K-R1/split_round1.json
runs=/data/runs
mkdir -p $runs/$name
enc=/data/init/$ENC
python experiments/K-M1/train_tsvad.py --data /data/oy16 --split-file $split --init-model $enc --aug-root /data/aug \
  --steps ${STEPS:-8000} --amp ${EXTRA:-} --resume --out $runs/$name >> $runs/$name/log.txt 2>&1
grep -q "TRAINING DONE" $runs/$name/log.txt || { echo "training failed"; exit 1; }
link=${LINK:-'{"emb_w":8}'}
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --max-minutes 100000 --conversations 113 --model $enc --tsvad $runs/$name/best.pt --link-params $link"
$e --cache $runs/$name/cache --tag _$name --out $runs/$name/full_val.json > $runs/$name/full_val.log 2>&1
$e --cache $runs/$name/cache --aug-root /data/aug --tag _$name --out $runs/$name/full_val_rec.json > $runs/$name/full_val_rec.log 2>&1
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("best.pt", "manifest.json", "full_val.json", "full_val_rec.json"):
    p = f"$runs/$name/{f}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"k-v3-$name/{f}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "$name DONE"
