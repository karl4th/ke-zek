# Kezek architecture

Status: design draft (2026-11-01). The choice between the audio and the Mimi variant is made
by experiment K-T0.

## Task

```
mixture  x = s_A + s_B        mono, exactly two speakers
enrollment e_A                 5-30 s of speaker A alone, from elsewhere in the same recording
                               (or any other recording of A)
output   ŝ_A ≈ s_A             mono, same length and alignment as x
```

What a good output does, region by region:

| Region | Expected output |
|---|---|
| only A talks | A unchanged (no damage, no level change) |
| only B talks | silence |
| both talk | A alone, as clean as possible: the hard case and the primary metric |
| neither talks | silence |

## Variants

### Audio (waveform in, waveform out)

A target speaker extraction network: a speaker encoder turns the enrollment into an
embedding, and an extractor conditioned on it estimates the target waveform or a mask. We
start from an open pretrained model (WeSep, SpeakerBeam family) and fine-tune it on our
mixtures. This is the expected quality leader and the reference for the Mimi variant.

### Mimi (latents in, tokens out)

- Input: continuous Mimi encoder latents of the mixture (512-d, before quantization,
  12.5 Hz) plus an embedding of the enrollment.
- Output: the 8 codebook tokens of the target, as Mimi would encode the clean target. They
  are decoded back to audio by the Mimi decoder.
- Why: 12.5 Hz instead of 16-24 kHz makes it orders of magnitude cheaper. For downstream
  uses that consume Mimi tokens anyway, the output is ready to use.
- Risk: the Mimi encoding of a mixture may already have lost detail of the quieter voice.
  Continuous latents lose less than tokens, so they are the input.
- Ceiling: the clean target passed through Mimi (encode → decode). No Mimi output can beat
  it, so all arms are scored after a Mimi round-trip to compare like with like.

## Enrollment

- Solo speech of the target (the other speaker silent), 5-30 s, chosen by
  `kezek.enrollment.pick_enrollment`.
- In evaluation it never overlaps the scored segment.
- At inference on real recordings without separate channels, solo stretches have to be found
  first (voice activity plus speaker change detection), or the user supplies the sample.

## Training data

Overlaid single-speaker clips (the answer is known exactly), built by `kezek.mixing`:

1. **Same-recording pairs.** Both speakers' clips come from one recording (same room,
   microphone and processing), so the model must separate by voice and not by acoustics.
2. **Conversation-like placement.** Turns alternate with gaps, short overlaps (0.2-2 s) and
   backchannels, instead of two clips fully on top of each other. Real conversation is ~10%
   overlap, so a model trained only on full overlap sees the wrong conditions.
3. **Also full-overlap pairs** (classic Libri2Mix style) as a smaller share, because they are
   the hardest case for learning to separate.
4. **Same processing.** Loudness matched, and the same codecs as the test sources (MP3,
   Zoom-style compression).

## Testing

Only real conversations recorded with one speaker per channel, summed to mono. Never test on
overlaid clips: those numbers look good and say little.

## Out of scope for now

More than two speakers, extraction without enrollment, streaming.
