# Kezek

**Target speaker extraction: given a mono recording of two people talking and a short sample of one speaker's voice, return mono audio with only that speaker.**

*Kezek* (Kazakh *кезек*: turn, one's turn to speak).

```
mono recording (speakers A + B)  ─┐
                                   ├─►  Kezek  ─►  mono recording of A only
voice sample of A (5-30 s)       ─┘
```

> **Status: pre-release, research in progress.** No model weights or results yet. The
> repository holds the design, the scoring code and the experiment protocols. Numbers will
> appear here only after they are measured.

## Scope

- **Input:** a mono mixture with exactly two speakers, and an enrollment clip (voice sample)
  of the target speaker. The enrollment is required.
- **Output:** mono audio of the target speaker, sample-aligned with the input. It should be
  silent where only the other speaker talks, unchanged where only the target talks, and clean
  where both talk at once.
- **Not in scope (for now):** more than two speakers, extraction without an enrollment,
  streaming.

## Approach

Two variants are compared in the first experiment ([K-T0](experiments/K-T0/README.md)):

| Variant | Works on | Output |
|---|---|---|
| Audio | waveform of the mixture + enrollment | waveform of the target |
| Mimi | continuous latents of the Mimi codec (12.5 Hz) + enrollment | Mimi tokens of the target, decoded to audio |

The audio variant starts from an open pretrained extraction model and is fine-tuned. The
Mimi variant is much cheaper, so it is chosen if it comes within 2 dB of the audio variant.

**Training data** is built by overlaying single-speaker clips, so the right answer is known
exactly. Clips are placed like a real conversation (turns, short overlaps, backchannels), and
both speakers' clips come from the same recording, so a model cannot separate by room or
microphone instead of by voice. **Testing** is done only on real conversations recorded
with one speaker per channel, with the channels summed to mono. See
[docs/architecture.md](docs/architecture.md).

## Languages

English first. Kazakh and Kazakh-Russian code-switched speech next: training mixtures from
Kazakh single-speaker audio, and a test set of real two-channel Kazakh conversations.

## Metrics

Scoring regions come from voice activity on the clean reference channels.

| Metric | Region | Meaning |
|---|---|---|
| **SI-SDR overlap** (primary) | both talk | how cleanly the target is recovered where it is hardest |
| SI-SDRi overlap | both talk | improvement over the unprocessed mixture |
| SI-SDR solo | only the target talks | the target must not be damaged |
| Leakage | only the other talks | share of frames where the output is audibly active |
| Residual (dB) | only the other talks | output energy relative to the other speaker (0 dB = passed through) |

Definitions: [docs/benchmark.md](docs/benchmark.md). Implementation:
[`src/kezek/metrics.py`](src/kezek/metrics.py).

## Quick start

```bash
git clone https://github.com/karl4th/ke-zek && cd ke-zek
pip install -e ".[dev]"
pytest

# Make a test item from a real two-channel conversation (one speaker per channel)
kezek prepare conversation.wav --target-channel 0 --out items/conv01_a

# ... run a model, write items/conv01_a/estimate.wav ...

# Score
kezek score items/* --json report.json
```

## Repository layout

| Path | Contents |
|---|---|
| `src/kezek/` | `metrics.py` (SI-SDR by region, leakage), `mixing.py` (training mixtures), `enrollment.py`, `vad.py`, `cli.py` |
| `docs/` | [architecture](docs/architecture.md), [roadmap](docs/roadmap.md), [benchmark](docs/benchmark.md), [data](docs/data.md), [experiment protocol](docs/experiment-protocol.md) |
| `experiments/` | One folder per experiment: protocol and outcome-to-action table, written before launch |
| `configs/` | Experiment configs |
| `tests/` | Unit tests |

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| K-T0 | Audio vs Mimi extraction; quick test on real two-channel English conversations | protocol drafted |
| K-T1 | Scale training data (conversation-style mixtures from podcasts); test on CANDOR | planned |
| K-T2 | Kazakh / Kazakh-Russian: training mixtures and a recorded two-channel test set | planned |
| K-T3 | Benchmark release and paper on arXiv | planned |

See [docs/roadmap.md](docs/roadmap.md).

## Related work

SpeakerBeam (Žmolíková et al., IEEE JSTSP 13(4), 2019),
SpEx+ ([arXiv:2005.04686](https://arxiv.org/abs/2005.04686)),
WeSep ([arXiv:2409.15799](https://arxiv.org/abs/2409.15799)),
SI-SDR ([arXiv:1811.02508](https://arxiv.org/abs/1811.02508)),
LibriMix ([arXiv:2005.11262](https://arxiv.org/abs/2005.11262)),
Mimi / Moshi ([arXiv:2410.00037](https://arxiv.org/abs/2410.00037)).

## Citation

A paper is planned for arXiv. Until then, please cite the repository; see
[CITATION.cff](CITATION.cff).

## License

Code: [MIT](LICENSE). Audio is never redistributed. Some evaluation and training corpora are
research-only, and weights trained on them are released only if their terms allow it (see
[docs/data.md](docs/data.md)).
