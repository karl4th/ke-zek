# K-E0: Mimi latents vs WavLM-Base+ under the segmentation head

Status: protocol draft. Depends on K0 (KezekBench v0 must exist and be frozen).
Config: [configs/k-e0.yaml](../../configs/k-e0.yaml).

## Questions

1. Which encoder does Kezek use: Mimi latents or WavLM-Base+?
2. Do Mimi latents carry speaker identity, i.e. is an embedding head on them worth building?

## Hypothesis

Mimi latents (512-d, pre-quantization, 12.5 Hz) carry enough speaker information that a
powerset segmentation head on them is at most 1 DER point worse than the same head on
WavLM-Base+ (mean-pooled to 12.5 Hz), on KezekBench v0.

## Arms

| Arm | Features | Rate |
|---|---|---|
| mimi | Mimi encoder latents before quantization, frozen | 12.5 Hz |
| wavlm | WavLM-Base+, one layer chosen on dev before launch, frozen, 4:1 mean-pooled | 50 → 12.5 Hz |

Same head (5-10M parameters, ±5% between arms), same data, same batch order, same steps.
Two seeds per arm.

## Data

- Train: pseudo-labels on 500-1000 h of podcasts (frames where pyannote 3.1 and DiariZen
  agree) plus AMI and VoxConverse dev; synthetic overlaps in 10% of batches.
- Test: KezekBench v0. VoxConverse test is reported for comparability only.
- No show in KezekBench appears in training.

## Metrics

- **Primary:** DER on KezekBench v0, local segmentation only (oracle speaker mapping inside
  each window, no clustering), so the encoder is compared and the clustering is not.
- **Guardrails:** leakage, boundary error p90, overlap recall.
- **Speaker probe (question 2):** linear classifier of speaker identity on frozen features
  (both encoders), on held-out speakers, frame-level accuracy against chance.

## Outcome → action

| Outcome | Action |
|---|---|
| DER(mimi) ≤ DER(wavlm) + 1.0 for both seeds, guardrails within bounds | Encoder = Mimi; one pass shared with codec encoding |
| DER(mimi) > DER(wavlm) + 1.0 | Encoder = WavLM-Base+; Kezek caches its own features |
| Anything in between (seeds disagree, guardrail blocks) | Grey zone → Mimi (simpler pipeline) |
| Speaker probe on Mimi ≈ chance | Embeddings from frozen ECAPA/WeSpeaker; no own embedding head |
| Speaker probe on Mimi clearly above chance | K1 adds a branch: own embedding head on the latents |

Guardrail bounds (to fix before launch): leakage and boundary p90 of the mimi arm no worse
than wavlm by more than 1 point / 20 ms.

## Preparation

- [ ] KezekBench v0 frozen (K0)
- [ ] Mimi latent cache (pre-quantization), exported during codec encoding
- [ ] WavLM-Base+ feature cache; layer chosen on dev
- [ ] Pseudo-labels: pyannote 3.1 + DiariZen on 500-1000 h, agreement mask
- [ ] Head type (Conformer / Mamba-2) and step count fixed in the config

## Record

- Commit of protocol and scoring code: TBD
- Results: `RESULT.md` (after the run)
