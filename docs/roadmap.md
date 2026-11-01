# Roadmap

Experiments are designed one stage at a time, based on the previous result. Each stage ends at
a gate: go, stop, or scale up. The gate is written down before the stage starts. A grey zone
always resolves to the simpler option. The rules are in
[experiment-protocol.md](experiment-protocol.md).

## K0: baselines and KezekBench v0 (now, RTX 3060)

- Collect 50 English podcast episodes from the DuplexChat manifest (they already have
  two-speaker labels and source links), plus 10 episodes from outside DuplexChat.
- Run pyannote 3.1, DiariZen, Sortformer and Silero VAD. Measure speed (hours of audio per
  hour on a 3060).
- **KezekBench v0:** 5 h of hand-annotated podcasts (turn boundaries ±50 ms, overlaps, speaker
  ids). Test only, never used for training or tuning. A second annotator re-checks 1 h. See
  [kezekbench.md](kezekbench.md).
- Score every baseline: DER and the duplex metrics. This is the reference point and the
  first table of the paper.
- **Gate:** if the best baseline already has leakage < 2% and boundary error p90 < 100 ms,
  Kezek v1 starts as that baseline plus the export step for the first thousands of hours.
  The own head is then trained in parallel for speed, not quality.

Protocol: [experiments/K0](../experiments/K0/README.md).

## K-E0: Mimi latents vs WavLM-Base+ (first experiment)

- The same powerset head (±5% parameters) on (a) Mimi latents (512-d, 12.5 Hz) and (b)
  WavLM-Base+ mean-pooled to 12.5 Hz. Trained on 500-1000 h of pseudo-labels plus AMI and
  VoxConverse.
- Primary metric: DER on KezekBench v0. Guardrails: leakage, boundary error, and a linear
  speaker probe on the frozen features.
- Decision: Mimi if its DER is no worse than WavLM's by more than 1 point absolute; otherwise
  WavLM. Grey zone → Mimi (simpler pipeline).

Protocol: [experiments/K-E0](../experiments/K-E0/README.md).

## K1: the head at scale

- Pseudo-labels for 5-10k h of podcasts (teachers plus DuplexChat labels), the head on the
  chosen encoder, and synthetic overlaps.
- Embeddings and clustering: ECAPA + AHC as in pyannote. A head on the latents only if K-E0
  shows they carry speaker identity.
- **Gate:** leakage ≤ best teacher, boundary error ≤ teacher, speed ≥ 10× teacher. Otherwise
  repeat K1 with a larger head or more data, and use teacher output downstream meanwhile.

## K2: pseudo-duplex export

- Filters: two main speakers ≥ 85% of speech, confidence, overlap → silence plus a flag,
  segments ≥ 60 s.
- Listen to 100 segments and compare conversation statistics with DuplexChat.
- Output: a manifest plus two channels as Mimi tokens.
- **Gate:** at least 98 of 100 segments have no audible B in channel A, and conversation
  statistics are within ±30% of DuplexChat.

## K3: KezekBench v1 and the paper

- Extend the test set to 10-20 h across three domains: English podcasts, Kazakh and
  Kazakh-Russian code-switched conversation, and open sets for comparability.
- All baselines plus Kezek. Release open weights, code and manifests (no audio). Post to arXiv.

## Later

- Fine-tune on Kazakh / Kazakh-Russian if the baselines degrade there. This is checked in K3
  before deciding.
- A streaming version.
