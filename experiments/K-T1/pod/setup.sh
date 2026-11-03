#!/bin/bash
# Prepare a RunPod pod (image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) for K-T1.
# Pod env: "GITHUB_API": "{{ RUNPOD_SECRET_GITHUB_API }}", "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
# "AIRON_API_KEY": "{{ RUNPOD_SECRET_AIRON_API_KEY }}". Clone the repository to /workspace/ke-zek first.
# Data goes to the container disk (/root/d): LibriSpeech ~60 GB, MUSAN ~11 GB, RIRS_NOISES ~1.3 GB,
# DNS Freesound noise ~4.5 GB compressed.
set -x
mkdir -p /root/d/dl /root/d/extracted && cd /root/d || exit 1
printf 'torch==2.4.1\ntorchaudio==2.4.1\nhuggingface_hub==0.25.2\nnumpy<2.1\n' > constraints.txt
apt-get update -qq && apt-get install -y -qq aria2 libsndfile1 ffmpeg unzip > /dev/null
pip install -q torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q -c constraints.txt soundfile hyperpyyaml sentencepiece joblib tqdm scipy packaging airon
pip install -q --no-deps speechbrain==1.0.3
pip install -q -c constraints.txt huggingface_hub==0.25.2
pip install -q -c constraints.txt --no-deps -e /workspace/ke-zek
python -c "import torch; torch.zeros(1).cuda(); print('GPU OK', torch.cuda.get_device_name(0), torch.__version__)"
ffmpeg -hide_banner -encoders | grep -E "libmp3lame|libopus"
for f in dev-clean train-clean-100 train-clean-360; do
  aria2c -q -x16 -s16 -d dl "https://openslr.elda.org/resources/12/$f.tar.gz" && tar -xzf "dl/$f.tar.gz" -C extracted --skip-old-files
done
aria2c -q -x16 -s16 -d dl https://openslr.elda.org/resources/17/musan.tar.gz && tar -xzf dl/musan.tar.gz -C extracted --skip-old-files
aria2c -q -x16 -s16 -d dl https://openslr.elda.org/resources/28/rirs_noises.zip && unzip -q -n dl/rirs_noises.zip -d extracted
# DNS Challenge noise: the Freesound part only (CC0). AudioSet and DEMAND are not used.
DNS=https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband
mkdir -p extracted/dns_freesound
for i in 000 001; do
  aria2c -q -x16 -s16 -d dl "$DNS/datasets_fullband.noise_fullband.freesound_$i.tar.bz2" && \
    tar -xjf "dl/datasets_fullband.noise_fullband.freesound_$i.tar.bz2" -C extracted/dns_freesound --skip-old-files
done
python -c "from huggingface_hub import hf_hub_download as d; print(d('manifestro/kezek-k-t0c', 'kezek_tse.pt', local_dir='/root/d/init'))"
find extracted/LibriSpeech -name '*.flac' | wc -l
find extracted/musan/noise extracted/musan/music extracted/dns_freesound -name '*.wav' | wc -l
find extracted/RIRS_NOISES/simulated_rirs extracted/RIRS_NOISES/real_rirs_isotropic_noises -name '*.wav' | wc -l
echo SETUP_DONE
