# Kezek v1 architecture

Status: accepted design (2026-11-01). The encoder choice is settled by experiment K-E0.
The goal is a fast, reliable pipeline from a mono podcast to per-speaker activity on Mimi
frames to pseudo-duplex training data. A state-of-the-art DER is not the goal.

## Pipeline

```
  Mono podcast (16/24 kHz, 2-4 speakers, up to hours)
        │
        ▼
  Encoder: Mimi latents (512-d, pre-quantization, 12.5 Hz), frozen,
           ONE pass shared with downstream codec encoding        ◄── alternative: WavLM-Base+ (K-E0)
        │
        ▼
  Local segmentation: head of 5-15M params (Conformer or Mamba-2), ~10 s overlapping windows,
           powerset output: silence / speaker k / pair of speakers      ◄── trained by distillation from
           (VAD, local speakers and overlap from one softmax)                pyannote 3.1 + DiariZen, DuplexChat
        │                                                                   labels, some human labels,
        ▼                                                                   synthetic overlaps
  Episode-level clustering: per-segment speaker embeddings (frozen ECAPA/WeSpeaker,
           or a head on the same latents if K-E0 shows they carry speaker identity), AHC / VBx
           → speaker ids consistent across the episode
        │
        ▼
  Activity: [speakers × frames] at 12.5 Hz + overlap mask + per-frame confidence
        │
        ▼
  Pseudo-duplex export: two main speakers → two channels (speech / silence),
           quality filters → manifest for the dialogue model
```

## Components

| Component | Does | Choice | Why |
|---|---|---|---|
| Encoder | audio → frame features | Mimi latents, frozen | Every hour is encoded with Mimi for the dialogue model anyway. One pass serves both, and the output is already on the right frame grid. Risk: little speaker information in the latents (the first codebook is distilled towards WavLM semantics). K-E0 tests this. Fallback: WavLM-Base+, 50 Hz mean-pooled to 12.5 Hz |
| Local segmentation | per window: who speaks in each frame, where overlap is | powerset over 3 local speakers, max 2 simultaneous (7 classes); 5-15M head | One output gives VAD, speakers and overlap. Powerset distils well. The size is set by the speed target of over 1000x real time on an RTX 3060 |
| Embeddings + clustering | episode-level speaker ids | frozen ECAPA/WeSpeaker + AHC first (as in pyannote 3.1); own embedding head only if K-E0 supports it | Known, cheap, nothing to invent |
| Activity | final matrix | 12.5 Hz, plus overlap and confidence (agreement across windows and teachers) | Export filters need confidence |
| Export | pseudo-duplex | the two main speakers must hold ≥ 85% of speech; overlap → silence in both channels plus a flag; low-confidence or third-speaker frames excluded; only runs ≥ 60 s | Diarization errors become echo and false turn timing in training. Dropping them is cheaper than learning them |

## Training the segmentation head

1. **Pseudo-labelled podcasts (bulk).** Run pyannote 3.1 and DiariZen on thousands of hours.
   Keep frames where both teachers agree on activity and on the number of speakers. Mask the
   rest out of the loss.
2. **DuplexChat labels.** Two-speaker labels for 282k h of English podcasts
   (pyannote-community-1) serve as an extra teacher at no cost.
3. **Human annotation.** AMI, VoxConverse, CALLHOME, DIHARD if available. This is a small
   share, so the student does not inherit the teachers' systematic errors.
4. **Synthetic data, only for overlap.** Turns spliced with 0.2-2 s overlap, so that the
   overlap classes are represented. Pseudo-labels contain little overlap, and it is unreliable.
5. **Loss.** Cross-entropy over powerset classes, with the disagreement mask and up-weighted
   overlap classes.

## Why a student can beat its teachers here

- It is an order of magnitude faster: a 5-15M head on cached latents, against WavLM-Large
  plus a Conformer.
- Its output is native to Mimi frames, with no resampling.
- It gives per-frame confidence and export filters, which teachers do not provide.
- Agreement across an ensemble of teachers is usually more accurate than any single teacher.
- It is trained in-domain. The teachers were trained on meetings and telephone speech, and the
  largest gain is expected on podcasts.

## Checks before the first export

- Listen to 100 random pseudo-duplex segments. Turn boundaries should match the labels by ear
  (±100 ms), and B should not be audible in channel A.
- Leakage on KezekBench.
- Compare conversation statistics against DuplexChat (6.2 turn changes/min, 3.1
  backchannels/min, 10% overlap). A large deviation means systematic diarization error.

## Out of scope for v1

- Streaming (a small 10-30M encoder with under 500 ms latency), if a product needs it.
- Separating overlapped speech: overlaps are detected and flagged, never separated.
- Speech recognition and language identification.
