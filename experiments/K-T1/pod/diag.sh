#!/bin/bash
# K-T1 diagnostic before the full run (after setup.sh): does overlap quality on clean mixtures
# come back during fine-tuning? The first 20k steps of the seed-0 K-T1 run exactly (the LR
# schedule spans 100k, the run stops at 20k), a checkpoint every 2.5k, then each checkpoint on
# the 300 K-T0c dev-clean mixtures (paired with K-T0c) and the validation set.
# Results go to $O/diag/*.json and to the Hugging Face repo $REPO (diag/ folder).
. /workspace/ke-zek/experiments/K-T0/pod/env.sh
cd /root/d || exit 1
export AIRON_URL=https://api.manifestro.ci OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=${O:-kt1-diag}
STEPS=${STEPS:-20000}
REPO=${REPO:-manifestro/kezek-k-t1-diag}
K=/workspace/ke-zek/experiments/K-T1
BANK="--rir-dirs extracted/RIRS_NOISES/simulated_rirs extracted/RIRS_NOISES/real_rirs_isotropic_noises \
  --noise-dirs extracted/musan/noise extracted/dns_freesound --music-dirs extracted/musan/music"
mkdir -p $O/diag
python $K/train.py --init /root/d/init/kezek_tse.pt \
  --train-roots extracted/LibriSpeech/train-clean-100 extracted/LibriSpeech/train-clean-360 \
  --val-root extracted/LibriSpeech/dev-clean $BANK \
  --steps 100000 --stop-at $STEPS --bs 8 --eval-every 2500 --keep-every 2500 --seed 0 --workers 16 --fast \
  --out $O --run-name k-t1-diag-${STEPS} >> $O/log.txt 2>&1
echo "=== train exit $? $(date -Is)" >> $O/log.txt
python $K/evaluate.py --checkpoint /root/d/init/kezek_tse.pt --final \
  --dev-clean-300 extracted/LibriSpeech/dev-clean --out $O/diag/k-t0c_dc300.json >> $O/log.txt 2>&1
for c in $(ls $O/step_*.pt | sort -V); do
  s=$(basename $c .pt)
  python $K/evaluate.py --checkpoint $c --final --dev-clean-300 extracted/LibriSpeech/dev-clean \
    --baseline $O/diag/k-t0c_dc300.json --out $O/diag/${s}_dc300.json >> $O/log.txt 2>&1
done
python $K/evaluate.py --checkpoint /root/d/init/kezek_tse.pt --val-root extracted/LibriSpeech/dev-clean $BANK \
  --out $O/diag/k-t0c_val.json >> $O/log.txt 2>&1
python $K/evaluate.py --checkpoint $O/kezek_tse.pt --val-root extracted/LibriSpeech/dev-clean $BANK \
  --out $O/diag/final_val.json >> $O/log.txt 2>&1
python - <<EOF >> $O/log.txt 2>&1
import glob, json, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("$REPO", private=True, exist_ok=True)
api.upload_folder(folder_path="$O/diag", path_in_repo="diag", repo_id="$REPO")
for f in ["log.txt", "eval_log.json", "manifest.json"]:
    api.upload_file(path_or_fileobj=f"$O/{f}", path_in_repo=f, repo_id="$REPO")
for f in sorted(glob.glob("$O/diag/step_*_dc300.json")):
    d = json.load(open(f))["dev_clean_300"]
    print("DIAG", os.path.basename(f), round(d["sisdri"], 2), round(d["vs_baseline"]["delta"], 2), d["vs_baseline"]["delta_ci"])
EOF
echo "DIAG DONE $(date -Is)" >> $O/log.txt
