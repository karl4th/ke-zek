# Open Yap training run

Use an extracted Open Yap WAV delivery with this layout:

```text
DATA_ROOT/conversations/conv_<id>/meta.json
DATA_ROOT/conversations/conv_<id>/speaker_a.wav
DATA_ROOT/conversations/conv_<id>/speaker_b.wav
```

The two tracks must be mono, sample aligned, and have different speaker IDs in
`meta.json`. The loader reads short windows from disk and resamples them to 16 kHz.
It makes both target directions from each pair. Enrollment comes from a separate
time span of the target track. Connected speaker groups stay in one data split.

On a training machine with AWS access, prepare a small slice first:

```bash
bash experiments/V2/prepare_open_yap.sh "$OPEN_YAP_S3_PREFIX" /data/open-yap 8
```

The script checks each tar against `SHA256SUMS`, extracts it, and removes only
the temporary local tar. Running it again skips completed parts. To prepare all
parts, pass `116` as the final argument. The source archives in S3 are retained.

Install the package with its Python dependencies, PyTorch, the Airon client,
and `huggingface_hub`. Training requires `AIRON_URL`, `AIRON_API_KEY`, `HF_TOKEN`, and an
existing private model repository for checkpoints. Keep training audio on the
training machine; the checkpoint uploader sends only model files and the split
manifest to the private repository.

```bash
python experiments/V2/train.py \
  --data-root /data/open-yap \
  --out /data/kezek-v2-run \
  --steps 1000 --batch-size 4 --eval-every 250 \
  --max-train-conversations 64 --max-val-conversations 16 \
  --hf-repo OWNER/PRIVATE_MODEL_REPO --require-clean
```

For a short throughput check that keeps the same schedule and can resume,
add `--stop-at 50`. Continue with `--resume` and omit `--stop-at`.

Each evaluation records target fidelity, overlap separation, and leakage in
Airon. `audio/step_<N>/` holds mixture, clean target, and estimate WAVs for
listening. The untouched test split is recorded in `manifest.json`. The run
writes `last.pt` for resuming and `model.pt` for inference; checkpoints are
uploaded after each evaluation and verified by remote file size.

To extract from a new mono recording after training:

```bash
python experiments/V2/infer.py \
  --checkpoint /data/kezek-v2-run/model.pt \
  --mixture input.wav --enrollment voice_sample.wav --output selected_speaker.wav
```

Inference accepts mono WAVs at their native sample rate, runs the model at
16 kHz in overlapping windows, and writes an output with the input duration.
