# Data

Kezek never redistributes audio. The repository holds manifests (ids, source links,
licenses, checksums) and the code to fetch and process audio from the original sources. Each
source keeps its own license.

## Test data: real two-channel conversations

| Source | Content | Size | License / access |
|---|---|---|---|
| Expresso (Meta) | improvised dialogues, 4 actors, studio, 48 kHz stereo, one actor per channel | ~30 h | CC BY-NC 4.0 (non-commercial) |
| CANDOR (BetterUp) | video-call conversations, one participant per channel, Zoom compression and echo cancellation | 1656 conversations, 850+ h | research use, by application (requested 2026-11-01) |
| otoSpeech full-duplex | two-person English conversations, channel-separated, 48 kHz | ~105 h / ~280 h | non-commercial, gated; voice-identity uses prohibited |
| Kazakh / Kazakh-Russian | own recordings, one microphone per speaker | TBD | ours |

CANDOR splits are shared with the Dauys project, which uses the same sessions: Dauys test and
sealed-test sessions, and all sessions of their speakers, are never used for training here.

## Training data: single-speaker clips for overlaying

| Source | Content | Notes |
|---|---|---|
| English podcasts | solo stretches of each speaker within an episode | main source; both speakers from the same episode |
| Training split of the stereo corpora | each channel's solo speech, or the real mix | real overlaps; speakers disjoint from test |
| LibriSpeech / Libri-Light | read speech | small share, for speaker variety |
| KSC2, Kazakh podcasts | Kazakh and Kazakh-Russian speech | for K-T2 |

## Rules

- **Speakers in a test set never appear in training**, in any form. For podcasts this means
  whole shows are held out, since hosts recur across episodes.
- Overlaid mixtures are for training only.
- **Licenses:** results on research-only or non-commercial corpora can be published. Weights
  trained on them are released only if the terms allow it.

## Manifest format

JSON lines, one recording per line:

```json
{"id": "expresso-ex04-ex01-001", "source": "expresso", "url": "...", "license": "CC-BY-NC-4.0",
 "duration_s": 312.4, "language": ["en"], "channels": {"0": "ex04", "1": "ex01"},
 "sha256": "...", "split": "test"}
```

`split` is one of `train`, `dev`, `test`, `excluded`.
