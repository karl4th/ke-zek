# K0: baselines and KezekBench v0

Status: in preparation. Hardware: RTX 3060 12 GB.

## Question

What do open diarization systems achieve on conversational podcasts, measured with DER and
the duplex metrics? Is any of them already good enough to build pseudo-duplex data from?

## Steps

1. **Episodes.** Pick 50 English episodes from the DuplexChat manifest and 10 from outside it,
   with 2-4 speakers each and varied shows, recording setups and lengths. Add their shows to
   `data/manifests/kezekbench_exclude.txt`.
2. **Annotation (KezekBench v0).** Hand-annotate 5 h following
   [docs/kezekbench.md](../../docs/kezekbench.md). Resolve the open guideline decisions first.
   A second annotator independently labels 1 h.
3. **Freeze.** Commit `kezekbench/v0/` (manifest, RTTM, UEM) and the scoring code. Record the
   commit hash here.
4. **Baselines.** Run on the KezekBench v0 audio:

   | System | Runner | Status |
   |---|---|---|
   | pyannote 3.1 | `scripts/run_pyannote.py` | runner written, not yet run |
   | pyannote community-1 (DuplexChat labels) | from manifest | TODO |
   | DiariZen | TODO | TODO |
   | NeMo Sortformer | TODO | TODO |
   | Silero VAD (VAD metrics only) | TODO | TODO |
   | Human (second annotator, 1 h) | - | TODO |

5. **Score.** `kezek evaluate --ref kezekbench/v0/reference.rttm --uem kezekbench/v0/test.uem
   --hyp runs/k0/<system>.rttm --json runs/k0/<system>.json`, and measure speed.

## Primary metric

Leakage on KezekBench v0 (it decides the gate). Reported alongside: DER (with no collar and
with a 250 ms collar), boundary error median/p90, overlap P/R, VAD F1, speed.

## Outcome → action

| Outcome (best baseline) | Action |
|---|---|
| Leakage < 2% **and** boundary p90 < 100 ms | Kezek v1 = that baseline plus the export step for the first thousands of hours; own head trained in parallel for speed |
| Otherwise | Own head (K-E0 → K1) is needed for quality as well as speed; the best baseline becomes the teacher reference |

## Record

- KezekBench v0 commit: TBD
- Results: `RESULT.md` (after the run)
