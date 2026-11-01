# KezekBench

A test set and a scoring protocol for diarization of conversational podcasts. Systems are
judged on standard DER and on what their output would do to two-channel (duplex) training
data.

**KezekBench is a test set only.** It is never used for training, model selection or
hyper-parameter tuning.

## Versions

| Version | Content | Status |
|---|---|---|
| v0 | 5 h of English podcasts, hand-annotated (2-4 speakers per episode) | in preparation (K0) |
| v1 | 10-20 h: English podcasts; Kazakh and Kazakh-Russian code-switched conversation; open sets (VoxConverse, AMI) for comparability | planned (K3) |

## Files

```
kezekbench/<version>/
  episodes.jsonl   one line per episode: id, source URL, license, duration, language(s),
                   number of speakers, audio checksum (sha256 of the 16 kHz mono WAV we score)
  reference.rttm   speaker segments
  test.uem         scored regions (whole episode unless stated)
```

Audio is not redistributed. `scripts/` will provide a downloader that fetches each episode
from its source, converts it to 16 kHz mono WAV and checks the checksum.

## Annotation guidelines (v0)

Annotators label speech segments per speaker in a waveform/spectrogram editor (Label Studio
or Audacity labels).

- **Boundaries** are set to within ±50 ms of the start and end of audible speech, at a zoom
  where this is visible.
- **A segment** is continuous speech of one speaker. Pauses of 250 ms or longer inside a turn
  split it into separate segments.
- **Overlap**: when two speakers talk at once, both segments are annotated. Overlap is not a
  separate label but follows from the segments.
- **Backchannels** ("mm-hm", "yeah", "right") are speech of that speaker and are annotated,
  including when they overlap the other speaker.
- **Speaker ids** are consistent within an episode. Ids across episodes are unrelated.

Open decisions, to be fixed before annotation starts and recorded here:

- [ ] Laughter, breaths, coughs: speech or not? (Proposal: laughter while speaking is speech;
      isolated laughter, breaths and coughs are not.)
- [ ] Music, jingles, ads read by a host: scored or excluded through the UEM?
      (Proposal: host-read ads are speech; music-only parts are excluded in the UEM.)
- [ ] Minimum segment length (proposal: no minimum; one-word backchannels are kept).

Quality control: a second annotator independently labels 1 h. We report inter-annotator DER
and boundary error on that hour as the human ceiling.

## Metrics

Implemented in [`src/kezek/metrics`](../src/kezek/metrics), run with `kezek evaluate`. Scoring
uses a 10 ms grid. Speakers are mapped one-to-one with the Hungarian algorithm, maximising
co-active time.

| Metric | Definition |
|---|---|
| **DER** | (missed + false alarm + confusion) / reference speech time, NIST definition. Overlap is scored. No collar by default; results with a 250 ms collar are reported alongside for comparability with older work |
| **Boundary error** (median, p90, ms) | Reference turns are obtained by bridging pauses < 250 ms. For every reference turn onset (offset), the distance to the nearest onset (offset) of the mapped hypothesis speaker, capped at 1 s. *Missed boundaries* = share at the cap |
| **Leakage** | Frames where a hypothesis speaker is active, its mapped reference speaker is silent and another reference speaker talks, divided by all hypothesis speech frames. Unmapped hypothesis speakers leak on any reference speech |
| **Overlap P / R / F1** | Frames with ≥ 2 active speakers, reference vs hypothesis |
| **VAD P / R / F1** | Frames with any active speaker |
| **Speed** | Hours of audio processed per wall-clock hour, on RTX 3060 and RTX 4090, with encoder cost reported separately |

Confidence intervals: 95% bootstrap over episodes (1000 resamples).

The 250 ms merge gap and 1 s cap are proposed defaults and are frozen with KezekBench v0.
