"""Checks for target-conditioned source routing."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kezek.target_router import TargetRouter, oracle_route_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "V2"))
from train_target_router import speaker_features  # noqa: E402


def test_router_uses_enrollment_and_follows_source_swap():
    time = torch.arange(8_000) / 8_000
    first = 0.1 * torch.sin(2 * torch.pi * 220 * time)
    second = 0.1 * torch.sin(2 * torch.pi * 440 * time)
    sources = torch.stack((first, second), dim=-1)[None]
    router = TargetRouter(embedding_dim=2, channels=8)
    voice = torch.tensor([[1.0, 0.0]])
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    estimate, logits = router(sources, voice, embeddings)
    reversed_estimate, _ = router(sources.flip(-1), voice, embeddings.flip(1))

    assert estimate.shape == (1, 8_000)
    assert logits.shape[1] == 3
    assert torch.mean((estimate - first) ** 2) < torch.mean((estimate - second) ** 2)
    assert torch.allclose(estimate, reversed_estimate, atol=1e-5)


def test_oracle_route_labels_include_target_and_silence():
    target = torch.cat((torch.full((64,), 0.1), torch.zeros(64)))[None]
    other = torch.full((128,), 0.2)
    sources = torch.stack((target[0], other), dim=-1)[None]

    labels = oracle_route_labels(sources, target, frames=2)

    assert labels.tolist() == [[0, 2]]


def test_speaker_features_accept_two_source_batches():
    class SimpleClassifier:
        def encode_batch(self, waveform):
            return torch.ones(1, 1, 192, device=waveform.device)

    time = torch.arange(8_000) / 8_000
    first = 0.1 * torch.sin(2 * torch.pi * 220 * time)
    second = 0.1 * torch.sin(2 * torch.pi * 440 * time)
    sources = torch.stack((first, second), dim=-1).repeat(2, 1, 1)
    enrollment = first.repeat(2, 2)

    voice, source_voices = speaker_features(SimpleClassifier(), enrollment, sources, "cpu")

    assert voice.shape == (2, 192)
    assert source_voices.shape == (2, 2, 192)
