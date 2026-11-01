# K-T0: target speaker extraction, audio vs Mimi latents

Status: protocol draft (2026-11-01). Runs in parallel with K0. Diarization is not needed:
enrollment is taken from the reference channels (oracle).
Config: [configs/k-t0.yaml](../../configs/k-t0.yaml).

## Idea

If one speaker's voice can be extracted cleanly from a two-person mono mix, building a stereo
conversation is trivial: extract A, extract B, put them in two channels. Unlike
cut-and-silence pseudo-duplex, this keeps overlaps and backchannels ("mm-hm" over the
partner's speech).

## Questions

1. Can target speaker extraction (TSE) recover one speaker from a real two-person mix cleanly
   enough to build stereo training data, overlaps included?
2. Can extraction on continuous Mimi latents, with Mimi tokens as output, get close to
   audio-domain extraction at a fraction of the cost?

## Ground truth

Real two-channel conversations (one speaker per channel) are summed to mono. The input is the
mix and the target is one original channel. Enrollment is 10-30 s of the target speaker's solo
speech, taken from another part of the same session.

## Arms

| Arm | Input | Output | Training |
|---|---|---|---|
| `cut` | mix + reference activity of the target | mix where only the target talks, silence elsewhere and in overlaps | none (current pseudo-duplex plan with oracle diarization) |
| `audio-zs` | mix waveform + enrollment | target waveform | pretrained open TSE model (WeSep or SpeakerBeam), zero-shot |
| `audio-ft` | as `audio-zs` | target waveform | `audio-zs` fine-tuned on our training mixes |
| `mimi` | continuous Mimi latents of the mix (512-d, pre-quantization, 12.5 Hz) + enrollment | Mimi tokens (8 codebooks) of the target channel | trained from scratch on our training mixes |
| `ceiling` | the clean target channel | Mimi round-trip of it | - |

All outputs go through Mimi (encode → decode, 24 kHz) before scoring. This puts every arm in
the form the dialogue model consumes. `ceiling` is the best any arm can reach.

## Data

- **Quick test (now):** Expresso improvised dialogues (stereo, one actor per channel, ~30 h,
  4 speakers, CC BY-NC 4.0). Also the DuplexChat E0b set from Dauys (101 clips), with the
  caveat that its channels are themselves separation output (DialogueSidon), not true ground
  truth.
- **Main test:** CANDOR held-out sessions (one channel per participant), using the same
  session/speaker splits as Dauys. The Dauys sealed test sessions are never used here.
- **Training mixes (`audio-ft`, `mimi`):** CANDOR training split summed to mono, plus
  same-episode overlays from podcasts (two solo stretches of A and B from one episode, mixed
  with overlaps of 0.2-2 s, so room and microphone match).

## Metrics

Regions come from VAD on the reference channels: *overlap* (both talk), *target solo*,
*target silent* (only the other talks).

- **Primary:** SI-SDR (dB) of the output on overlap regions.
- **Guardrails:**
  - Leakage: share of target-silent frames where VAD on the output fires.
  - Preservation: SI-SDR on target-solo regions, compared with `ceiling`.
  - Backchannel retention: share of short target turns (< 1 s) inside partner speech that
    are present (VAD) in the output.
  - Speaker similarity: ECAPA cosine between output and enrollment.
  - Speed: hours of audio per hour on RTX 3060.

## Outcome → action (thresholds to fix before launch)

| Outcome | Action |
|---|---|
| Best audio arm: overlap SI-SDR ≥ 10 dB, leakage < 2%, solo SI-SDR ≥ ceiling − 1 dB, backchannel retention ≥ 80% | Stereo through TSE becomes the main path for Dauys stage 2: original audio in solo regions, TSE in overlaps (hybrid) |
| `mimi` within 2 dB of the best audio arm on overlap SI-SDR and passes the same guardrails | Use `mimi` (cheaper; one Mimi pass for everything) |
| No arm passes | Keep cut-and-silence (`cut`); TSE is not used |
| Grey zone (seeds disagree, a guardrail blocks) | The simpler option: `cut` over TSE, `mimi` over audio only if it passes outright |

Two seeds for trained arms (`audio-ft`, `mimi`).

## Preparation

- [ ] Download Expresso improvised dialogues; check channel mapping and crosstalk
- [ ] Reference VAD on each channel; region masks (overlap / solo / silent)
- [ ] Enrollment picker (solo stretches, other part of the session)
- [ ] Scoring script (SI-SDR by region, leakage, backchannel retention, speaker similarity)
- [ ] `audio-zs` with a pretrained TSE model, run on Expresso
- [ ] CANDOR access (requested 2026-11-01), splits shared with Dauys
- [ ] `mimi` arm: model and training script

## Licensing note

Expresso and CANDOR are research-only or non-commercial. Results can be published, but
weights trained on them cannot be released under MIT without checking the terms first.

## Record

- Commit of protocol and scoring code: TBD
- Results: `RESULT.md` (after the run)
