# Roadmap

Goal: from a mono recording of two people, each speaker on their own track (two mono or one
stereo file), overlaps included. Step 1 is extracting one speaker (mono → mono); step 2 runs
it for both speakers and builds the two tracks; step 3 is Kazakh, the benchmark and the paper.

Experiments are designed one stage at a time, based on the previous result. Each stage ends
at a gate that is written down before the stage starts. A grey zone always resolves to the
simpler option. Rules: [experiment-protocol.md](experiment-protocol.md).

## Step 1. K-T0: audio vs Mimi extraction (now, RTX 3060)

- Quick test on real two-channel English conversations (Expresso improvised dialogues),
  channels summed to mono, one channel as the target.
- Arms: pretrained audio model zero-shot, the same fine-tuned on our mixtures, a Mimi-latent
  model trained from scratch, and the Mimi round-trip ceiling.
- Primary metric: SI-SDR on overlap regions. Guardrails: leakage, SI-SDR on solo regions.
- Decision: which variant Kezek is built on.

Protocol: [experiments/K-T0](../experiments/K-T0/README.md).

## Step 1. K-T1: scale

- Training mixtures from thousands of hours of English podcasts: same-recording,
  conversation-like placement.
- Test on CANDOR held-out sessions (real video calls, one participant per channel).
- **Gate** (to fix after K-T0): overlap SI-SDR and leakage targets on CANDOR.

## Step 2. K-S: two speakers → two mono / stereo

- The same extractor run twice on one mixture, with the sample of A and the sample of B.
- On real mono recordings the voice samples have to be found automatically: stretches where
  each person talks alone. This needs a "who speaks when" step (diarization, or voice
  activity plus speaker change and clustering).
- Checked on real stereo: both channels against the reference, leakage between channels,
  overlaps and backchannels preserved.
- **Gate:** to fix after K-T1.

## Step 3. K-T2: Kazakh

- Training mixtures from Kazakh and Kazakh-Russian single-speaker audio (KSC2, Kazakh
  podcasts).
- Test set: real two-person Kazakh / Kazakh-Russian conversations recorded with one
  microphone per speaker. No such open corpus is known, so we record it.
- First check how much the English model loses on Kazakh, then decide on fine-tuning.

## Step 3. K-T3: benchmark and paper

- KezekBench: real two-channel conversations in English and Kazakh / Kazakh-Russian, with
  scoring code ([benchmark.md](benchmark.md)).
- All baselines and Kezek; open code and manifests; weights where the data terms allow.
  Paper on arXiv.
