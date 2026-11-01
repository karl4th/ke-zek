import numpy as np
import pytest

from kezek.powerset import IGNORE_INDEX, Powerset


def test_classes():
    ps = Powerset(3, 2)
    assert ps.classes == [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    assert ps.num_classes == 7
    assert Powerset(4, 2).num_classes == 1 + 4 + 6


def test_encode_decode_roundtrip():
    ps = Powerset(3, 2)
    rng = np.random.default_rng(0)
    act = rng.random((3, 500)) < 0.3
    act[:, act.sum(axis=0) > 2] = False  # drop triple overlaps
    ids = ps.encode(act)
    assert ids.min() >= 0
    np.testing.assert_array_equal(ps.decode(ids), act)


def test_too_many_speakers_is_ignored():
    ps = Powerset(3, 2)
    act = np.ones((3, 4), dtype=bool)
    assert (ps.encode(act) == IGNORE_INDEX).all()
    assert not ps.decode(ps.encode(act)).any()
    with pytest.raises(ValueError):
        ps.encode(act, strict=True)


def test_multilabel_marginals():
    ps = Powerset(2, 2)  # classes: {}, {0}, {1}, {0,1}
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_allclose(ps.to_multilabel(probs), [[0.6, 0.7]])
