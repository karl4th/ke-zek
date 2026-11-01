# Roadmap

Experiments are designed one stage at a time, based on the previous result. Each stage ends
at a gate that is written down before the stage starts. A grey zone always resolves to the
simpler option. Rules: [experiment-protocol.md](experiment-protocol.md).

## K-T0: audio vs Mimi extraction (now, RTX 3060)

- Quick test on real two-channel English conversations (Expresso improvised dialogues),
  channels summed to mono, one channel as the target.
- Arms: pretrained audio model zero-shot, the same fine-tuned on our mixtures, a Mimi-latent
  model trained from scratch, and the Mimi round-trip ceiling.
- Primary metric: SI-SDR on overlap regions. Guardrails: leakage, SI-SDR on solo regions.
- Decision: which variant Kezek is built on.

Protocol: [experiments/K-T0](../experiments/K-T0/README.md).

## K-T1: scale

- Training mixtures from thousands of hours of English podcasts: same-recording,
  conversation-like placement.
- Test on CANDOR held-out sessions (real video calls, one participant per channel).
- **Gate** (to fix after K-T0): overlap SI-SDR and leakage targets on CANDOR.

## K-T2: Kazakh

- Training mixtures from Kazakh and Kazakh-Russian single-speaker audio (KSC2, Kazakh
  podcasts).
- Test set: real two-person Kazakh / Kazakh-Russian conversations recorded with one
  microphone per speaker. No such open corpus is known, so we record it.
- First check how much the English model loses on Kazakh, then decide on fine-tuning.

## K-T3: benchmark and paper

- KezekBench: real two-channel conversations in English and Kazakh / Kazakh-Russian, with
  scoring code ([benchmark.md](benchmark.md)).
- All baselines and Kezek; open code and manifests; weights where the data terms allow.
  Paper on arXiv.
