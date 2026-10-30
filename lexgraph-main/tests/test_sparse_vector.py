"""Sparse vector algebra tests against dense reference computations."""

from __future__ import annotations

import math
import random

import pytest

from structures.sparse_vector import SparseVector


def _dense(vector: SparseVector) -> list[float]:
    dense = [0.0] * vector.dimension
    for index, value in vector:
        dense[index] = value
    return dense


def _random_vector(rng: random.Random, dimension: int, nnz: int) -> SparseVector:
    indices = sorted(rng.sample(range(dimension), nnz))
    return SparseVector(dimension, indices, [rng.uniform(-3, 3) for _ in indices])


def test_validation():
    with pytest.raises(ValueError):
        SparseVector(5, [0, 1], [1.0])
    with pytest.raises(ValueError):
        SparseVector(5, [2, 1], [1.0, 1.0])
    with pytest.raises(ValueError):
        SparseVector(5, [9], [1.0])


def test_from_pairs_sorts_and_merges():
    vector = SparseVector.from_pairs(10, [(3, 1.0), (1, 2.0), (3, 4.0), (5, 0.0)])
    assert vector.indices == [1, 3]
    assert vector.values == [2.0, 5.0]


def test_dot_matches_dense():
    rng = random.Random(1)
    for _ in range(20):
        a = _random_vector(rng, 60, 12)
        b = _random_vector(rng, 60, 15)
        expected = sum(x * y for x, y in zip(_dense(a), _dense(b)))
        assert a.dot(b) == pytest.approx(expected)


def test_norm_and_cosine_match_dense():
    rng = random.Random(2)
    a = _random_vector(rng, 40, 9)
    b = _random_vector(rng, 40, 11)
    da, db = _dense(a), _dense(b)
    assert a.norm() == pytest.approx(math.sqrt(sum(x * x for x in da)))
    assert a.norm_sq() == pytest.approx(sum(x * x for x in da))
    cosine = sum(x * y for x, y in zip(da, db)) / (
        math.sqrt(sum(x * x for x in da)) * math.sqrt(sum(y * y for y in db))
    )
    assert a.cosine(b) == pytest.approx(cosine)


def test_add_and_scale_match_dense():
    rng = random.Random(3)
    a = _random_vector(rng, 30, 8)
    b = _random_vector(rng, 30, 10)
    da, db = _dense(a), _dense(b)
    assert _dense(a.add(b)) == pytest.approx([x + y for x, y in zip(da, db)])
    assert _dense(a.scale(-2.5)) == pytest.approx([-2.5 * x for x in da])


def test_add_cancellation_drops_coordinate():
    a = SparseVector(5, [1], [2.0])
    b = SparseVector(5, [1], [-2.0])
    assert a.add(b).nnz == 0


def test_squared_distance_matches_dense():
    rng = random.Random(4)
    a = _random_vector(rng, 25, 7)
    b = _random_vector(rng, 25, 6)
    expected = sum((x - y) ** 2 for x, y in zip(_dense(a), _dense(b)))
    assert a.squared_distance(b) == pytest.approx(expected)


def test_normalised_distance_equals_cosine_identity():
    a = SparseVector(4, [0, 2], [0.6, 0.8])
    b = SparseVector(4, [1, 2], [0.6, 0.8])
    assert a.squared_distance(b) == pytest.approx(2 - 2 * a.cosine(b))


def test_dot_dense_matches_sparse():
    vector = SparseVector(6, [1, 4], [2.0, -1.0])
    dense = [0.5, 2.0, 3.0, 0.0, -1.0, 9.0]
    assert vector.dot_dense(dense) == pytest.approx(2.0 * 2.0 + (-1.0) * (-1.0))


def test_zero_vector_cosine_is_zero():
    assert SparseVector.zeros(5).cosine(SparseVector(5, [0], [1.0])) == 0.0


def test_norm_is_cached():
    vector = SparseVector(3, [0], [3.0])
    assert vector.norm() == pytest.approx(3.0)
    vector.values[0] = 100.0  # mutating after caching keeps the cached norm
    assert vector.norm() == pytest.approx(3.0)
