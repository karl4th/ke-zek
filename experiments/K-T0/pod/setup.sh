#!/bin/bash
# Prepare a RunPod pod (image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) for train_scale.py.
# Pod env: "GITHUB_API": "{{ RUNPOD_SECRET_GITHUB_API }}", "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
# "AIRON_API_KEY": "{{ RUNPOD_SECRET_AIRON_API_KEY }}". Clone the repository to /workspace/ke-zek first.
# Put the data on the container disk: /workspace is a 20 GB volume.
set -x
mkdir -p /root/d/dl /root/d/extracted && cd /root/d || exit 1
printf 'torch==2.4.1\ntorchaudio==2.4.1\nhuggingface_hub==0.25.2\nnumpy<2.1\n' > constraints.txt
apt-get update -qq && apt-get install -y -qq aria2 libsndfile1 > /dev/null
pip install -q torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q -c constraints.txt soundfile hyperpyyaml sentencepiece joblib tqdm scipy packaging airon
pip install -q --no-deps speechbrain==1.0.3
pip install -q -c constraints.txt huggingface_hub==0.25.2
pip install -q -c constraints.txt --no-deps -e /workspace/ke-zek
python -c "import torch; torch.zeros(1).cuda(); print('GPU OK', torch.cuda.get_device_name(0), torch.__version__)"
for f in dev-clean train-clean-100 train-clean-360; do
  aria2c -q -x16 -s16 -d dl "https://openslr.elda.org/resources/12/$f.tar.gz" && tar -xzf "dl/$f.tar.gz" -C extracted --skip-old-files
done
find extracted/LibriSpeech -name '*.flac' | wc -l
echo SETUP_DONE
