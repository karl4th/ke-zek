# K-T0: target speaker extraction, audio vs Mimi latents

Status: protocol draft (2026-11-01). First experiment of Kezek.
Config: [configs/k-t0.yaml](../../configs/k-t0.yaml).

## Task

Input: a mono recording of two people and 10 s of the target speaker's voice. Output: mono
audio with only the target speaker.

## Questions

1. How cleanly can one speaker be extracted from a **real** two-person conversation with an
   open extraction model, out of the box and after fine-tuning on our mixtures?
2. Does extraction on continuous Mimi latents (Mimi tokens out) come close to audio-domain
   extraction?

## Ground truth

Real conversations recorded with one speaker per channel, summed to mono; one channel is the
target. Each conversation gives two items (each speaker as the target). The enrollment is 10 s
of the target's solo speech from outside the scored segment (`kezek prepare`).

## Arms

| Arm | Input | Output | Training |
|---|---|---|---|
| `mixture` | - | the unprocessed mixture | floor (reference point for SI-SDRi) |
| `audio-zs` | mixture waveform + enrollment | target waveform | open pretrained model (WeSep or SpeakerBeam family), no fine-tuning |
| `audio-ft` | as `audio-zs` | target waveform | `audio-zs` fine-tuned on our mixtures |
| `mimi` | Mimi latents of the mixture (512-d, pre-quantization, 12.5 Hz) + enrollment | Mimi tokens (8 codebooks) of the target, decoded | trained from scratch on our mixtures |
| `ceiling` | the clean target | Mimi round-trip of the target | - |

All arms are scored after a Mimi round-trip (encode → decode, 24 kHz), so every output is
compared in the same form, and `ceiling` is the upper bound.

## Data

- **Test:** Expresso improvised dialogues, held-out sessions, both channels as targets.
  CANDOR is added when access arrives (K-T1).
- **Training mixtures** (`audio-ft`, `mimi`): overlaid single-speaker clips (`kezek.mixing`):
  both speakers from the same recording, conversation-like placement (alternating turns,
  20% of transitions overlapped by 0.2-2 s), plus 20% full-overlap pairs. Sources: solo
  stretches of English podcasts, and the non-test Expresso sessions *only if* their speakers
  do not appear in the test sessions. Expresso has only 4 speakers, so in practice it is
  test-only.
- Same training data, steps and batch order for `audio-ft` and `mimi`.

## Metrics

From [docs/benchmark.md](../../docs/benchmark.md):

- **Primary:** `sisdr_overlap`, SI-SDR on overlap regions.
- **Guardrails:** `leakage` (output audible where only the other speaker talks), `sisdr_solo`
  (target preserved where it talks alone).
- **Report only:** `sisdri_overlap`, `residual_db`, speaker similarity to the enrollment,
  speed on RTX 3060.

Two seeds for the trained arms.

## Outcome → action (thresholds to confirm before launch)

"Passes" means: `sisdr_overlap` ≥ 10 dB, `leakage` < 2%, `sisdr_solo` ≥ ceiling − 1 dB.

| Outcome | Action |
|---|---|
| `mimi` passes and is within 2 dB of the best audio arm on `sisdr_overlap`, both seeds | Kezek is built on Mimi latents |
| Best audio arm passes, `mimi` does not | Kezek is built on audio; Mimi is dropped |
| `audio-zs` already passes | Fine-tuning is not a priority; K-T1 scales data for robustness (CANDOR, Kazakh) instead |
| Nothing passes | K-T1 = larger audio model and more conversation-like training data; Mimi dropped |
| Grey zone (seeds disagree, one guardrail blocks) | The simpler option: audio over Mimi only if Mimi fails outright; otherwise Mimi |

## Preparation

- [ ] Download Expresso improvised dialogues; check channel mapping and crosstalk
- [ ] Fix the test sessions; `kezek prepare` for both channels of each
- [ ] Pick the pretrained model for `audio-zs`; run it; first numbers
- [ ] Mimi round-trip for all arms; `ceiling`
- [ ] Training mixtures from podcast solo stretches
- [ ] `mimi` model and training script
- [ ] Speaker similarity (ECAPA) and bootstrap CIs in the scoring report

## Record

- Commit of protocol and scoring code: TBD
- Results: `RESULT.md` (after the run)
