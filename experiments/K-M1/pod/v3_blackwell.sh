#!/usr/bin/env bash
# Kezek v3 stage 1 on one RTX PRO 6000 Blackwell instance (g7e.2xlarge: 8 vCPU, 96 GB GPU): picks up
# where the two A10G instances stopped (2026-11-07 13:53 UTC), from HF manifestro/kezek-k-r1:
#   * tsvad_enc (profile head trained with the encoder) continued from k-v3-tsvad_enc/last.pt
#     (step 8000 of 12000), then its full evaluation;
#   * meanwhile the ensemble of the finished heads (tsvad_sim3 + tsvad_ms3) on full recordings,
#     clean and rendered;
#   * at the end the ensemble with the finished tsvad_enc as well.
# Word-gated labels with 0.3 s gap closing. Results to HF k-v3-<name>/. Every run logs to the Airon tracker.
#   git clone -b kezek-r1 https://github.com/karl4th/ke-zek.git ~/ke-zek
#   sudo mkdir -p /data && sudo chown $USER /data
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_blackwell.sh > /data/v3_blackwell.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log 2>/dev/null || bash "$here/setup.sh" > ~/setup.log 2>&1
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export KEZEK_LABELS=words KEZEK_WORD_GAP=0.3
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # PyTorch >= 2.6 on Blackwell: checkpoints here hold plain dicts
export AIRON_EXPERIMENT="K-V3-S1W stage 1 on word-gated activity labels"
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import hf_hub_download as d
t = os.environ["HF_TOKEN"]
for f in ("k-v3-v3s1w_g03/best_snap.pt", "k-v3-tsvad_sim3/head.pt", "k-v3-tsvad_ms3/head.pt",
          "k-v3-tsvad_enc/last.pt", "k-v3-tsvad_enc/best.pt"):
    d("manifestro/kezek-k-r1", f, local_dir="/data/init", token=t)
print("MODELS READY")
PY
grep -q "AUG READY" /data/get_aug.log 2>/dev/null || ( bash "$here/get_aug.sh" > /data/get_aug.log 2>&1 ) &
for i in 1 2 3; do bash "$here/get_data.sh" /data/oy16 8 >> /data/get_data.log 2>&1 && break; done
wait
grep -q "AUG READY" /data/get_aug.log || { echo "aug download failed"; exit 1; }
[ "$(ls -d /data/oy16/conv_* | wc -l)" -ge 1100 ] || { echo "data incomplete"; exit 1; }
split=$repo/experiments/K-R1/split_round1.json
enc=/data/init/k-v3-v3s1w_g03/best_snap.pt
runs=/data/runs
mkdir -p $runs/tsvad_enc $runs/ens2 $runs/ens3
[ -e $runs/tsvad_enc/last.pt ] || cp /data/init/k-v3-tsvad_enc/last.pt /data/init/k-v3-tsvad_enc/best.pt $runs/tsvad_enc/
up() {  # hf-dir files...
  local dst=$1
  shift
  python - "$dst" "$@" <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sys.argv[2:]:
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"{sys.argv[1]}/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED", sys.argv[1])
PY
}
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --max-minutes 100000 --conversations 113 --model $enc --link-params {\"emb_w\":8}"

# 1. continue tsvad_enc (same arguments as on the A10G, resumes from last.pt)
( python experiments/K-M1/train_tsvad.py --data /data/oy16 --split-file $split --init-model $enc --aug-root /data/aug \
    --steps 12000 --amp --sim-scales 1.0:0.25,2.0:0.5 --layers 3 --bc-weight 15 --train-enc --lr-enc 3e-5 \
    --workers 4 --resume --out $runs/tsvad_enc >> $runs/tsvad_enc/log.txt 2>&1
  up k-v3-tsvad_enc $runs/tsvad_enc/best.pt $runs/tsvad_enc/last.pt $runs/tsvad_enc/log.txt ) &
# 2. meanwhile: ensemble of the two finished heads, clean and rendered
h2=/data/init/k-v3-tsvad_sim3/head.pt,/data/init/k-v3-tsvad_ms3/head.pt
$e --tsvad $h2 --cache $runs/cache_h8 --tag _ens2 --out $runs/ens2/clean.json > $runs/ens2/clean.log 2>&1 &
$e --tsvad $h2 --cache $runs/cache_h8 --aug-root /data/aug --tag _ens2 --out $runs/ens2/rec.json > $runs/ens2/rec.log 2>&1 &
wait
up k-v3-ens2 $runs/ens2/clean.json $runs/ens2/rec.json
grep -q "TRAINING DONE" $runs/tsvad_enc/log.txt || { echo "tsvad_enc training failed"; exit 1; }
# 3. the finished tsvad_enc alone, and the ensemble of all three
mkdir -p $runs/tsvad_enc/eval_h8i1
h3=$h2,$runs/tsvad_enc/best.pt
$e --tsvad $runs/tsvad_enc/best.pt --cache $runs/cache_h8 --tag _tsvad_enc --out $runs/tsvad_enc/eval_h8i1/clean.json > $runs/tsvad_enc/eval_h8i1/clean.log 2>&1 &
$e --tsvad $runs/tsvad_enc/best.pt --cache $runs/cache_h8 --aug-root /data/aug --tag _tsvad_enc --out $runs/tsvad_enc/eval_h8i1/rec.json > $runs/tsvad_enc/eval_h8i1/rec.log 2>&1 &
$e --tsvad $h3 --cache $runs/cache_h8 --tag _ens3 --out $runs/ens3/clean.json > $runs/ens3/clean.log 2>&1 &
$e --tsvad $h3 --cache $runs/cache_h8 --aug-root /data/aug --tag _ens3 --out $runs/ens3/rec.json > $runs/ens3/rec.log 2>&1 &
wait
up k-v3-tsvad_enc/eval_h8i1 $runs/tsvad_enc/eval_h8i1/clean.json $runs/tsvad_enc/eval_h8i1/rec.json
up k-v3-ens3 $runs/ens3/clean.json $runs/ens3/rec.json
echo "V3 BLACKWELL DONE"
