# KezekBench

A test set and scoring protocol for target speaker extraction on **real** two-person
conversations.

Each test item is a real conversation recorded with one speaker per channel. The channels
are summed to a mono mixture, one channel is the target, and an enrollment of the target is
taken from outside the scored segment. Overlaid (simulated) mixtures are never used for
testing.

**KezekBench is test-only.** It is never used for training, model selection or tuning, and no
speaker in it appears in training data.

## Planned content

| Part | Source | Status |
|---|---|---|
| English, studio | Expresso improvised dialogues (held-out sessions) | quick test for K-T0 |
| English, video calls | CANDOR held-out sessions | access requested |
| Kazakh / Kazakh-Russian | own recordings, one microphone per speaker | planned (K-T2) |

## Item layout

`kezek prepare conversation.wav --target-channel 0 --out <item>` creates:

```
<item>/
  mixture.wav      channels summed, scored segment only
  target.wav       target channel, scored segment
  interferer.wav   other channel, scored segment
  enrollment.wav   target solo speech from outside the scored segment (default 10 s)
  item.json        source, channel, segment, where the enrollment came from
```

A system writes `<item>/estimate.wav` (same rate and length as `mixture.wav`), and
`kezek score <items>` scores it.

Every conversation gives two items: one per speaker as the target.

## Regions

From energy voice activity on each clean reference channel (20 ms frames, threshold 40 dB
below the channel's 95th-percentile frame energy, 100 ms hangover):

- **overlap**: both channels active.
- **solo**: only the target channel active.
- **absent**: only the other channel active.

## Metrics

| Metric | Definition |
|---|---|
| `sisdr_overlap` (primary) | SI-SDR (dB) of the estimate against the target on overlap regions |
| `sisdri_overlap` | `sisdr_overlap` minus the SI-SDR of the unprocessed mixture on the same regions |
| `sisdr_solo` | SI-SDR on solo regions: is the target preserved? |
| `leakage` | share of absent frames where the estimate's frame energy exceeds the target channel's own activity threshold, i.e. something audible is output while only the other speaker talks |
| `residual_db` | estimate energy over interferer energy on absent regions (0 dB = passed through untouched) |
| `sisdr` | SI-SDR on the whole segment |

Also reported per system: speaker similarity between the estimate and the enrollment (ECAPA
cosine), and speed (hours of audio per hour on RTX 3060).

Aggregation: mean over items, with a 95% bootstrap confidence interval over conversations.

Known limitation: real recordings have some crosstalk (the other voice faintly audible in a
speaker's own microphone). The reference therefore contains a little of the interferer. It will
be measured per source on absent regions of the reference itself and reported.
