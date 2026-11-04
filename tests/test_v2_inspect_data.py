"""Word timing diagnostics use the same time base as a sampled audio window."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")


def test_word_activity_tracks_window_offset(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "V2" / "inspect_data.py"
    spec = importlib.util.spec_from_file_location("v2_inspect", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    path = tmp_path / "transcript.json"
    path.write_text(
        json.dumps({"words": [{"start": 1.04, "end": 1.08}, {"start": 2.0, "end": 2.1}]})
    )
    activity = module._word_activity(path, start=1.0, seconds=0.2)
    assert len(activity) == 10
    assert activity.tolist() == [False, False, True, True, False, False, False, False, False, False]
