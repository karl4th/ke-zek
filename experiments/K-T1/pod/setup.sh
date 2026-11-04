#!/bin/bash
# Prepare a RunPod pod (image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) for K-T1.
# Pod env: "GITHUB_API": "{{ RUNPOD_SECRET_GITHUB_API }}", "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
# "AIRON_API_KEY": "{{ RUNPOD_SECRET_AIRON_API_KEY }}". Clone the repository to /workspace/ke-zek first.
# Data goes to the container disk (/root/d), about 55 GB once extracted: LibriSpeech dev-clean,
# train-clean-100/360, MUSAN, RIRS_NOISES, DNS Freesound noise. Archives are extracted as they
# download and not kept, so an 80 GB container disk is enough.
set -x
. /workspace/ke-zek/experiments/K-T0/pod/env.sh  # secrets live only in the container's first process
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

# The DNS archives stream straight into tar while the openslr ones download, so the network never
# waits for an extraction. A finished archive leaves a marker in extracted/.done, so a second run
# skips it; a broken stream is fetched again from the start (tar overwrites files).
apt-get install -y -qq pigz lbzip2 > /dev/null
mkdir -p extracted/.done extracted/dns_freesound
stream() {  # url, decompressor, target dir
  local url=$1 dec=$2 dir=$3 name=$(basename "$1")
  [ -e extracted/.done/$name ] && return 0
  for try in 1 2 3; do
    if [ -e dl/$name ] && [ ! -e dl/$name.aria2 ]; then  # a whole archive from an earlier run
      $dec -dc dl/$name | tar -x -C $dir && rm -f dl/$name && touch extracted/.done/$name && return 0
    elif curl -sSfL --retry 5 "$url" | $dec -dc | tar -x -C $dir; then
      touch extracted/.done/$name && return 0
    fi
    rm -f dl/$name dl/$name.aria2
    echo "stream $name failed, try $try"
  done
  return 1
}
fetch() {  # url, extract command: openslr gives ~5 MB/s per connection, so 16 connections to a file
  local url=$1 name=$(basename "$1"); shift
  [ -e extracted/.done/$name ] && return 0
  aria2c -q -x16 -s16 -d dl "$url" && "$@" dl/$name && rm -f dl/$name && touch extracted/.done/$name
}
tgz() { pigz -dc "$1" | tar -x -C extracted; }
zp() { unzip -q -o "$1" -d extracted; }
set -o pipefail
O=https://openslr.elda.org/resources
# one archive at a time: train-clean-360 and its extraction together nearly fill the 80 GB disk
( fetch $O/28/rirs_noises.zip zp && fetch $O/17/musan.tar.gz tgz \
  && for f in dev-clean train-clean-100 train-clean-360; do fetch $O/12/$f.tar.gz tgz || exit 1; done ) &
# DNS Challenge noise: the Freesound part only (CC0). AudioSet and DEMAND are not used.
DNS=https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband
for i in 000 001; do
  stream "$DNS/datasets_fullband.noise_fullband.freesound_$i.tar.bz2" lbzip2 extracted/dns_freesound &
done
wait
n=$(ls extracted/.done | wc -l)
[ "$n" -eq 8 ] || { echo "only $n of 8 archives extracted"; exit 1; }
python -c "from huggingface_hub import hf_hub_download as d; print(d('manifestro/kezek-k-t0c', 'kezek_tse.pt', local_dir='/root/d/init'))"
find extracted/LibriSpeech -name '*.flac' | wc -l
find extracted/musan/noise extracted/musan/music extracted/dns_freesound -name '*.wav' | wc -l
find extracted/RIRS_NOISES/simulated_rirs extracted/RIRS_NOISES/real_rirs_isotropic_noises -name '*.wav' | wc -l
df -h /root/d
echo SETUP_DONE
