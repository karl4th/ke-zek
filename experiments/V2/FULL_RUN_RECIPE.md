# Full training decision: target voice extraction

## Decision

Train the direct target voice extractor in `train_seprq_tse.py`. Its encoder stays
frozen; the extraction head learns from aligned, natural two-speaker mixtures.
The input is a mono mixture and a separate 15-second sample of the requested
voice. The output is one mono signal of that voice, including speech during
overlap. The reference for every training window is the original isolated track.

The recipe is based on the speaker-disjoint Open Yap split already recorded in
the run manifest: 1,392 training, 117 validation, and 93 test conversations.
No test conversation is used for checkpoint selection. Enrollment is from a
separate time span of the target track, at least two seconds away from the
mixture crop. The training loader presents both requested speakers for each
mixture and samples around transcript overlap half the time.

## Evidence

All figures below use 4-second, 16 kHz windows with mixture-aligned output
gain. Higher overlap SI-SDRi and short-turn improvement are better; lower
leakage is better. The short-turn measure uses transcript-defined target turns
of at most 1.2 seconds during overlap and reports reduction in waveform error
against the unprocessed mixture.

| Split | Windows | Head | Overlap SI-SDRi | Short-turn improvement | Leakage |
| --- | ---: | --- | ---: | ---: | ---: |
| Validation | 512 | Initial | +1.23 dB | +5.25 dB (43) | 0.462 |
| Validation | 512 | 2,000 steps | +1.81 dB | +5.94 dB (43) | 0.447 |
| Test | 512 | Initial | +4.30 dB | +8.71 dB (49) | 0.365 |
| Test | 512 | 2,000 steps | +4.68 dB | +9.46 dB (49) | 0.336 |

The test comparison uses paired windows. Across 90 conversations with a valid
overlap score, the mean gain from adaptation is +0.38 dB, with a 95% bootstrap
interval of +0.17 to +0.63 dB when resampling conversations. The validation
and test windows are sampled, not full-length recordings. This does not establish
perceptual quality or production readiness.

The previous two-output separation and speaker-selection route achieved an
oracle separation score of +6.49 dB at 20,000 steps but selected the requested
voice poorly: -3.04 dB overlap SI-SDRi on the 128-window validation comparison.
The direct route is therefore the full-run choice.

## Full run

Start from the pinned public checkpoint loaded by `train_seprq_tse.py`, or
resume from the 2,000-step local checkpoint if it has been retained. Use the
same speaker-disjoint split manifest throughout. The 2,000-step pilot took
about seven minutes on the experiment machine, including five evaluations of
128 windows. An evaluation of 512 windows took about 52 seconds. At the same
measured throughput, a 20,000-step run with 512-window checks every 500 steps
should take roughly 1.5 hours, about $1.60 of compute at the recorded hourly
rate. This is an estimate, not a reservation or a guaranteed duration. The
20,000-step cap is an experiment limit, not a quality target.

```bash
export PYTHONPATH=/home/ubuntu/seprq-addons
export HF_HOME=/home/ubuntu/seprq-cache
python experiments/V2/train_seprq_tse.py \
  --data-root /home/ubuntu/open-yap \
  --split-manifest /home/ubuntu/sepformer-scale20k/manifest.json \
  --out /home/ubuntu/direct-extraction-full \
  --steps 20000 --batch-size 2 --lr 3e-6 \
  --eval-every 500 --val-examples 512 \
  --enrollment-seconds 15 \
  --run-name v2-direct-extraction-full
```

The command requires the configured Airon credentials on the training machine.
It writes local `step_XXXXXX.pt` checkpoints and validation reports at each
evaluation. Resume with `--resume PATH` and the same settings if interrupted.
No model or audio is uploaded by this script.

After the capped run, select the checkpoint by validation overlap SI-SDRi, subject to short-turn
improvement remaining at least as high as the initial model and leakage not
exceeding the initial model by more than 0.03. Review both target
directions, short target turns, target-only speech, and absent-target regions.
Evaluate the selected checkpoint on the untouched test split once, using
`evaluate_seprq.py --split test --head-checkpoint PATH`. Listen to complete
recordings before a production release; window metrics alone cannot verify
backchannel preservation or natural sound.

## Run records

- Pilot head adaptation: Airon run `v2-seprq-tse-head-2k-low-lr`.
- Expanded validation: Airon run `0a995049-4867-4ae0-acd4-008fceeeb661`.
- Held-out test, adapted: Airon run `4138fe67-debc-469f-81a0-cf3d2e21e5c1`.
- Held-out test, initial: Airon run `24826a3c-d2a5-42e6-bfb3-c3249503e625`.

The 2,000-step head checkpoint is currently local to the experiment machine.
Its SHA-256 is
`723b8eb2e8874746266915335307ca7386f717c0f160649ddab22ee29470b94a`.
Retain it or use the initial public checkpoint when launching the full run.
