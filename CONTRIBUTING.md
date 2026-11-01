# Contributing

## Setup

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

CI runs the same checks on Python 3.10 and 3.12.

## Rules

- **No audio, features or checkpoints in git.** Commit manifests and code only
  (see [docs/data.md](docs/data.md)).
- **KezekBench is test-only.** Do not train, tune or select models on it, and keep its
  speakers out of all training data.
- **Test only on real two-channel recordings.** Overlaid mixtures are for training.
- **Experiments follow the protocol** in
  [docs/experiment-protocol.md](docs/experiment-protocol.md): the protocol, its outcome →
  action table and the scoring code are committed before training starts.
- **Metric changes** need a test that pins the new behaviour, and a note in
  [docs/benchmark.md](docs/benchmark.md). Once a KezekBench version is frozen, its metrics
  change only with a new version.
- Code, comments and documentation are in English.
