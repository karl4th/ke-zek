# K-T1: conversation windows and domain augmentation

Fine-tunes the final K-T0c checkpoint on simulated two-person conversations (turns, pauses,
short and full overlaps, backchannels) with a channel per window (reverb, telephone band,
noise, music, gain, clipping, codec), with the loss in `kezek.losses.kt1_loss`.

## Code

| What | Where |
|---|---|
| Conversation simulator, frame labels, L_ref | `src/kezek/mixing.py` (`simulate_conversation`, `ConversationConfig`) |
| Domain chain | `src/kezek/augment.py` (`AugmentConfig`, `sample_chain`, `Chain`) |
| Loss: SNR with a 30 dB ceiling on active frames, silence penalty on the residual, speaker term | `src/kezek/losses.py` |
| Metrics by segment type | `src/kezek/metrics.py` (`segment_scores`, `aggregate`, `wrong_speaker`) |
| Inference in 4 s windows with 1 s overlap | `src/kezek/infer.py` |
| Data, training, evaluation | `experiments/K-T1/kt1_data.py`, `train.py`, `evaluate.py` |
| Code checks | `tests/test_kt1.py`, `experiments/K-T1/overfit_check.py` |

Parameters are the defaults of `ConversationConfig`, `AugmentConfig` and the `train.py` flags.

## Run

```bash
pip install -e ".[dev]" scipy speechbrain && python -m pytest -q   # needs ffmpeg for the codec tests
python experiments/K-T1/overfit_check.py --init kezek_tse.pt --root .../dev-clean
python experiments/K-T1/train.py --init kezek_tse.pt \
    --train-roots .../train-clean-100 .../train-clean-360 --val-root .../dev-clean \
    --rir-dirs .../RIRS_NOISES/simulated_rirs .../RIRS_NOISES/real_rirs_isotropic_noises \
    --noise-dirs .../musan/noise .../dns_freesound --music-dirs .../musan/music --out run
python experiments/K-T1/evaluate.py --checkpoint run/kezek_tse.pt --val-root .../dev-clean   # validation
python experiments/K-T1/evaluate.py --checkpoint run/kezek_tse.pt --final --tests tests.json  # tests
```

On a pod: `pod/setup.sh`, then `pod/run.sh` next to `experiments/K-T0/pod/guard.py`
(`--keep-pod` keeps the pod after the final upload). `train.py` exits with code 3 when the
target-solo level on validation is off by more than 3 dB twice in a row from step 10k;
`run.sh` does not restart it.

Test lists are read only with `--final`. MUSAN `speech` is refused as background.

Background sources and licenses: MUSAN noise and music (CC BY 4.0), the Freesound part of the
DNS Challenge noise (CC0 1.0), RIRS_NOISES (Apache 2.0).
