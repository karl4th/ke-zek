"""Checks for target-conditioned source routing."""

import pytest

torch = pytest.importorskip("torch")

from kezek.target_router import TargetRouter, oracle_route_labels


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
