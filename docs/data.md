# Data

Kezek never redistributes audio. The repository holds manifests (episode ids, source links,
licenses, checksums, labels) and the code to fetch and process the audio from the original
sources. Each source keeps its own license, and users are responsible for complying with it.

## Sources

| Source | Use | Size | Labels | License / access |
|---|---|---|---|---|
| English podcasts (DuplexChat manifest, arXiv 2607.04941) | pseudo-label training (K-E0, K1), KezekBench v0 episodes | 282k h listed | two-speaker labels from pyannote-community-1 | per episode; check before use |
| English podcasts outside DuplexChat | KezekBench v0 (10 episodes), domain check | small | hand annotation | per episode |
| AMI Meeting Corpus | human-labelled training share | ~100 h | manual | CC BY 4.0 |
| VoxConverse | human-labelled training share (dev), comparability (test) | ~64 h | manual | CC BY 4.0 |
| CALLHOME (LDC) | optional human-labelled share | ~17 h | manual | LDC license |
| DIHARD III | optional, if access is granted | ~34 h | manual | LDC license |
| Kazakh / Kazakh-Russian conversation | KezekBench v1, possible fine-tuning | TBD | hand annotation | TBD; collection starts next |

## Test / train separation

- KezekBench episodes, **and every other episode of the same show**, are excluded from all
  training data. Podcast hosts recur across episodes, so separating by episode alone would
  leak speakers.
- Exclusions are kept in `data/manifests/kezekbench_exclude.txt` (show ids) and checked by
  the data tooling.

## Manifest format

JSON lines, one episode per line:

```json
{"id": "dc-000123", "source": "duplexchat", "url": "https://...", "show_id": "...",
 "license": "...", "duration_s": 3605.2, "language": ["en"], "num_speakers": 2,
 "sha256": "...", "split": "train"}
```

`split` is one of `train`, `dev`, `test`, `excluded`.

## Pseudo-labels

Teacher outputs are stored as RTTM per teacher and source, plus an agreement mask. Frames
where the teachers disagree on activity or on the number of speakers are masked out of the
loss, not relabelled. The teachers' versions and settings are recorded with every
pseudo-label set.
