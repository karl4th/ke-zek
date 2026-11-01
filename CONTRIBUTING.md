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
- **KezekBench is test-only.** Do not train, tune or select models on it, and do not add any
  episode of a KezekBench show to training data.
- **Experiments follow the protocol** in
  [docs/experiment-protocol.md](docs/experiment-protocol.md): the protocol, its outcome →
  action table and the scoring code are committed before training starts.
- **Metric changes** need a test that pins the new behaviour, and a note in
  [docs/kezekbench.md](docs/kezekbench.md). Once a KezekBench version is frozen, its metrics
  change only with a new version.
- Code, comments and documentation are in English.
