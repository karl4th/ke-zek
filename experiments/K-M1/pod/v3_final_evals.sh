#!/usr/bin/env bash
# Final stage 1 evaluations on the Blackwell instance, in parallel shards: once tsvad_enc has
# finished training, tsvad_enc alone and the ensemble of all three heads (tsvad_sim3, tsvad_ms3,
# tsvad_enc), full recordings of the 113 validation conversations, clean and rendered (rendered
# audio from HF manifestro/kezek-render-cache when it is there), two shards each; shards merged.
# Results to HF k-v3-tsvad_enc/eval_h8i1/ and k-v3-ens3/. Every run logs to the Airon tracker.
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_final_evals.sh > /data/v3_final_evals.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore KEZEK_PACK_SOLO=1
export KEZEK_LABELS=words KEZEK_WORD_GAP=0.3 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export AIRON_EXPERIMENT="K-V3-S1W stage 1 on word-gated activity labels"
runs=/data/runs
until grep -q "TRAINING DONE" $runs/tsvad_enc/log.txt; do sleep 30; done
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi, snapshot_download
t = os.environ["HF_TOKEN"]
api = HfApi(token=t)
for f in ("best.pt", "last.pt", "log.txt"):
    api.upload_file(path_or_fileobj=f"$runs/tsvad_enc/{f}", path_in_repo=f"k-v3-tsvad_enc/{f}", repo_id="manifestro/kezek-k-r1")
try:
    snapshot_download("manifestro/kezek-render-cache", repo_type="dataset", local_dir="/data/xcache", token=t)
    print("XCACHE", len(os.listdir("/data/xcache/render_all")))
except Exception as e:
    print("no render cache yet:", type(e).__name__)
PY
split=$repo/experiments/K-R1/split_round1.json
enc=/data/init/k-v3-v3s1w_g03/best_snap.pt
h3=/data/init/k-v3-tsvad_sim3/head.pt,/data/init/k-v3-tsvad_ms3/head.pt,$runs/tsvad_enc/best.pt
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --max-minutes 100000 --conversations 113 --model $enc --link-params {\"emb_w\":8} --cache $runs/cache_h8 --x-cache /data/xcache/render_all --shards 2"
mkdir -p $runs/fe/enc $runs/fe/ens3
for s in 0 1; do
  $e --shard $s --tsvad $runs/tsvad_enc/best.pt --tag _tsvad_enc_s$s --out $runs/fe/enc/clean_$s.json > $runs/fe/enc/clean_$s.log 2>&1 &
  $e --shard $s --tsvad $runs/tsvad_enc/best.pt --aug-root /data/aug --tag _tsvad_enc_s$s --out $runs/fe/enc/rec_$s.json > $runs/fe/enc/rec_$s.log 2>&1 &
  $e --shard $s --tsvad $h3 --tag _ens3_s$s --out $runs/fe/ens3/clean_$s.json > $runs/fe/ens3/clean_$s.log 2>&1 &
  $e --shard $s --tsvad $h3 --aug-root /data/aug --tag _ens3_s$s --out $runs/fe/ens3/rec_$s.json > $runs/fe/ens3/rec_$s.log 2>&1 &
done
wait
m=experiments/K-M1/merge_shards.py
for n in enc ens3; do
  for c in clean rec; do
    python $m --name s1_full_val_${c}_$n --out $runs/fe/$n/$c.json $runs/fe/$n/${c}_0.json $runs/fe/$n/${c}_1.json > $runs/fe/$n/$c.log 2>&1
  done
done
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for n, dst in (("enc", "k-v3-tsvad_enc/eval_h8i1"), ("ens3", "k-v3-ens3")):
    for c in ("clean", "rec"):
        api.upload_file(path_or_fileobj=f"$runs/fe/{n}/{c}.json", path_in_repo=f"{dst}/{c}.json", repo_id="manifestro/kezek-k-r1")
print("UPLOADED")
PY
echo "FINAL EVALS DONE"
