#!/usr/bin/env bash
# One-time setup of a fresh Airon AWS instance (Ubuntu, CUDA driver): venv with the pinned
# stack, the repository installed in place, MUSAN + RIRS_NOISES in the background.
set -e
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv libsndfile1 aria2 ffmpeg unzip pigz > /dev/null 2>&1
sudo mkdir -p /data && sudo chown "$USER" /data
python3 -m venv ~/venv
~/venv/bin/pip install -q --upgrade pip
~/venv/bin/pip install -q numpy scipy soundfile
~/venv/bin/pip install -q --no-deps -e ~/ke-zek
# Blackwell GPUs (sm_120, e.g. RTX PRO 6000 on g7e) need CUDA 12.8 builds; A10G and earlier keep 2.4.1
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE "blackwell|rtx pro 6000|b200|b100"; then
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
else
  ~/venv/bin/pip install -q torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
fi
~/venv/bin/pip install -q hyperpyyaml sentencepiece joblib tqdm packaging huggingface_hub==0.25.2 "numpy<2.1"
~/venv/bin/pip install -q --no-deps speechbrain==1.0.3
~/venv/bin/pip install -q airon
~/venv/bin/python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
echo SETUP DONE
