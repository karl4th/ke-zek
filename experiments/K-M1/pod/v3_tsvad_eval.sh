#!/usr/bin/env bash
# Kezek v3 stage 1 with the profile head: full recordings of the 113 validation conversations,
# clean and rendered (word-gated labels), with a chosen window hop and number of profile passes.
# Results to HF k-v3-$NAME/eval_$TAG/. Every run logs to the Airon tracker.
#   NAME=tsvad_sim HEAD=/data/runs/tsvad_sim/best.pt ENC=/data/init/k-v3-v3s1w_g03/best_snap.pt \
#     HOP=4 ITERS=2 TAG=h4i2 nohup bash ~/ke-zek/experiments/K-M1/pod/v3_tsvad_eval.sh > /data/eval_$TAG.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export KEZEK_LABELS=words KEZEK_WORD_GAP=${WORD_GAP:-0.3}
export AIRON_EXPERIMENT="K-V3-S1W stage 1 on word-gated activity labels"
out=/data/runs/$NAME/eval_$TAG
mkdir -p $out
link=${LINK:-'{"emb_w":8}'}
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $repo/experiments/K-R1/split_round1.json --max-minutes 100000 --conversations 113 --model $ENC --tsvad $HEAD --tsvad-iters ${ITERS:-1} --hop-s ${HOP:-8} --link-params $link --cache /data/runs/$NAME/cache_h${HOP:-8}"
$e --tag _${NAME}_$TAG --out $out/clean.json > $out/clean.log 2>&1 &
$e --aug-root /data/aug --tag _${NAME}_$TAG --out $out/rec.json > $out/rec.log 2>&1 &
wait
python - <<PY
import glob, os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("$out/*.json")):
    api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-$NAME/eval_$TAG/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
api.upload_file(path_or_fileobj="$HEAD", path_in_repo=f"k-v3-$NAME/head.pt", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "EVAL $NAME $TAG DONE"
