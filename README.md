# Kezek

**Fast speaker diarization for podcasts on Mimi latents, and KezekBench, a benchmark that scores diarization by what it does to two-channel conversation data.**

*Kezek* (Kazakh *кезек*: turn, one's turn to speak) finds who speaks when in long mono
recordings and turns them into pseudo-duplex conversations: speaker A in one channel,
speaker B in the other. Data like this is needed to train full-duplex spoken dialogue models,
and stereo recordings of real conversations are scarce. Mono podcasts are not scarce.

> **Status: pre-release, research in progress.** No model weights or benchmark results yet.
> The repository holds the design, the evaluation code and the experiment protocols. The
> roadmap is below. Numbers will appear here only after they are measured.

## Why

Open diarization systems are tuned for meetings and telephone speech, and they are scored with
DER. When diarization output is used to build two-channel training data, DER does not tell
you whether that data will be usable. A late turn boundary leaks the partner's voice into a
speaker's channel. A missed overlap shows up as false interruption timing. Kezek targets that
use:

- **Fast.** A small segmentation head (5-15M parameters) runs on frozen audio-codec latents
  (Mimi, 12.5 Hz). The latents are computed once and shared with the downstream dialogue
  model. The target is over 1000x real time on a single RTX 3060, so tens of thousands of hours
  can be processed in days.
- **Aligned to the codec frame grid.** The output is per-speaker activity at 12.5 Hz, the Mimi
  frame rate, with no resampling or re-alignment.
- **Calibrated.** Per-frame confidence and quality filters decide what gets exported.
  Uncertain material is dropped, not exported.
- **Measured on duplex metrics.** KezekBench reports turn-boundary error, cross-speaker leakage
  and overlap recall alongside DER.

## Languages

- **English** podcasts: the first target and the first KezekBench test set.
- **Kazakh, and Kazakh-Russian code-switched conversation:** next. A KezekBench test set is
  being prepared. Fine-tuning will follow if baselines fall short there.

Kezek does not use transcripts and is not tied to a language. Language matters for evaluation
and for the training-data mix.

## Approach

```
mono episode ─► Mimi encoder (frozen, shared) ─► segmentation head (powerset, 10 s windows)
            ─► speaker embeddings + clustering ─► per-speaker activity @ 12.5 Hz + overlap + confidence
            ─► pseudo-duplex export (two main speakers, quality filters)
```

The segmentation head is distilled from an ensemble of open teachers (pyannote 3.1,
DiariZen). Only frames where the teachers agree are used. A smaller share of human-annotated
data (AMI, VoxConverse) and synthetic overlaps are added. Design rationale:
[docs/architecture.md](docs/architecture.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/kezek/` | Python package: frame grid, RTTM I/O, powerset encoding, metrics, pseudo-duplex export, CLI |
| `docs/` | [architecture](docs/architecture.md), [roadmap](docs/roadmap.md), [KezekBench](docs/kezekbench.md), [data](docs/data.md), [experiment protocol](docs/experiment-protocol.md) |
| `experiments/` | One folder per experiment, with its protocol and outcome-to-action table written before launch |
| `configs/` | Experiment configs |
| `scripts/` | Baseline runners and data tooling |
| `data/manifests/` | Episode manifests (links and metadata only, no audio) |
| `tests/` | Unit tests |

## Quick start

```bash
git clone https://github.com/karl4th/ke-zek && cd ke-zek
pip install -e ".[dev]"
pytest

# Score any diarization output against a reference (RTTM in, DER + duplex metrics out)
kezek evaluate --ref ref.rttm --hyp hyp.rttm [--uem test.uem] [--json report.json]
```

Run the pyannote baseline (needs a GPU and a Hugging Face token with access to the gated
pyannote models):

```bash
pip install -e ".[baselines]"
python scripts/run_pyannote.py --audio-list audio.txt --out runs/k0/pyannote31.rttm
```

## Metrics

| Metric | Meaning |
|---|---|
| DER (missed / false alarm / confusion) | NIST definition, overlap scored, no collar by default |
| Boundary error, median and p90 (ms) | Distance from each reference turn onset/offset to the nearest one of the mapped hypothesis speaker |
| Leakage | Share of a speaker's channel frames where only someone else is actually talking: this becomes echo in duplex training data |
| Overlap precision / recall | Frames with two or more speakers |
| VAD precision / recall | Frames with any speech |
| Speed | Hours of audio per hour on RTX 3060 / 4090 |

Definitions: [docs/kezekbench.md](docs/kezekbench.md). Implementation:
[`src/kezek/metrics`](src/kezek/metrics).

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| K0 | Baselines (pyannote 3.1, DiariZen, Sortformer, Silero VAD) and KezekBench v0: 5 h of hand-annotated English podcasts | in preparation |
| K-E0 | Encoder choice: Mimi latents vs WavLM-Base+ under the same head | protocol drafted |
| K-T0 | Target speaker extraction from a two-person mix: audio vs Mimi latents | protocol drafted |
| K1 | Segmentation head at scale (5-10k h of pseudo-labels) and clustering | planned |
| K2 | Pseudo-duplex export with quality filters, listening checks | planned |
| K3 | KezekBench v1 (English, Kazakh / Kazakh-Russian, open sets), paper on arXiv | planned |

Every stage has a go/stop gate that is written down before the stage starts. See
[docs/roadmap.md](docs/roadmap.md).

## Related work

pyannote 3.1 ([arXiv:2310.13025](https://arxiv.org/abs/2310.13025)),
DiariZen ([arXiv:2509.26177](https://arxiv.org/abs/2509.26177)),
VBx ([arXiv:2012.14952](https://arxiv.org/abs/2012.14952)),
Sortformer ([arXiv:2409.06656](https://arxiv.org/abs/2409.06656)),
WavLM ([arXiv:2110.13900](https://arxiv.org/abs/2110.13900)),
Mimi / Moshi ([arXiv:2410.00037](https://arxiv.org/abs/2410.00037)),
DuplexChat ([arXiv:2607.04941](https://arxiv.org/abs/2607.04941)).

## Citation

A paper is planned for arXiv. Until then, please cite the repository; see
[CITATION.cff](CITATION.cff).

## License

Code: [MIT](LICENSE). Model weights will be released under MIT when available. Audio is never
redistributed: manifests point to the original sources, which keep their own licenses (see
[docs/data.md](docs/data.md)).
