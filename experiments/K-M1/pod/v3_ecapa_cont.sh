#!/usr/bin/env bash
# Kezek v3 step 1b: ECAPA fine-tuning continued from step 4000 (HF k-v3-ecapa1/ecapa_last.pt, the
# run that ran out of RAM) to 8000; weights to HF k-v3-ecapa2/ right after training, then stage 1
# with packed solo audio and the new ECAPA on full held-out conversations and kezek-v1-bench.
# Runs next to v3_packft.sh on the same instance (shares /data). Every run logs to the Airon tracker.
#   nohup bash ~/ke-zek/experiments/K-M1/pod/v3_ecapa_cont.sh > /data/v3_ecapa_cont.log 2>&1 &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
grep -q "SETUP DONE" ~/setup.log || { echo "setup failed"; exit 1; }
cd "$repo"
export PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export AIRON_EXPERIMENT="K-V3-S1 stage 1: profiles and activity (WavLM + Conformer, 16 s)"
pip install -q pyarrow
python - <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import hf_hub_download as d
d("manifestro/kezek-k-r1", "k-v3-ecapa1/ecapa_last.pt", local_dir="/data/init", token=os.environ["HF_TOKEN"])
print("INIT READY")
PY
grep -q "PREP DONE" /data/vox_prep.log 2>/dev/null || \
  python experiments/K-M1/ecapa_ft.py --prep --vox /data/vox --shards 10 > /data/vox_prep.log 2>&1
grep -q "PREP DONE" /data/vox_prep.log || { echo "voxceleb prep failed"; exit 1; }
until grep -q "AUG READY" /data/get_aug.log 2>/dev/null; do sleep 30; done
mkdir -p /data/runs
python experiments/K-M1/ecapa_ft.py --vox /data/vox --aug-root /data/aug --out /data/runs/ecapa2 \
  --init /data/init/k-v3-ecapa1/ecapa_last.pt --start-step 4000 > /data/runs/ecapa2.log 2>&1
grep -q "TRAINING DONE" /data/runs/ecapa2.log || { echo "ecapa training failed"; exit 1; }
up() {
  python - "$@" <<PY
import os, sys
sys.path.insert(0, "experiments/K-R1")
import tracker
tracker.load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sys.argv[1:]:
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"k-v3-ecapa2/{os.path.basename(f)}", repo_id="manifestro/kezek-k-r1")
print("UPLOADED", len(sys.argv) - 1)
PY
}
up /data/runs/ecapa2/ecapa.pt /data/runs/ecapa2/ecapa_last.pt /data/runs/ecapa2/head_last.pt /data/runs/ecapa2.log
# the first evaluation (v3_packft.sh) has to be done with the GPU and the oy16 download
until grep -qE "V3 PACKFT DONE|failed" /data/v3_packft.log; do sleep 30; done
split=$repo/experiments/K-R1/split_round1.json
model=/data/init/k-v3-v3s1c/best.pt
mkdir -p /data/fe2
export KEZEK_PACK_SOLO=1 KEZEK_ECAPA=/data/runs/ecapa2/ecapa.pt
e="python experiments/K-M1/s1_eval_full.py --data /data/oy16 --split-file $split --conversations 113 --model $model --cache /data/fe2/cache"
$e --out /data/fe2/packft2_clean.json > /data/fe2/packft2_clean.log 2>&1 &
$e --aug-root /data/aug --out /data/fe2/packft2_rec.json > /data/fe2/packft2_rec.log 2>&1 &
bash "$here/s1_bench.sh" /data/fe2/bench_packft2.json packft2=$model > /data/fe2/bench_packft2.log 2>&1 &
wait
for r in "" "--rec"; do
  echo "packft2 $r $(python experiments/K-M1/s1_swap_diag.py --data /data/oy16 --split-file $split --cache /data/fe2/cache $r 2>/dev/null | tail -1)"
done > /data/fe2/diag.txt
up /data/fe2/packft2_clean.json /data/fe2/packft2_rec.json /data/fe2/bench_packft2.json /data/fe2/diag.txt
echo "V3 ECAPA CONT DONE"
