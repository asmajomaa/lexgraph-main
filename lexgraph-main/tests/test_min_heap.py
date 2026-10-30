"""Heap tests against ``heapq`` and a full-sort oracle."""

from __future__ import annotations

import heapq
import random

import pytest

from structures.min_heap import MinHeap, TopKHeap, sort_top_k


def test_push_pop_order_matches_heapq_min_ordering():
    rng = random.Random(5)
    values = [rng.randint(-1000, 1000) for _ in range(300)]
    heap = MinHeap()
    for value in values:
        heap.push((float(value), value))
    expected = sorted((float(value), value) for value in values)
    popped = []
    while len(heap) > 0:
        item = heap.pop()
        assert item is not None
        popped.append(item)
    assert popped == expected


def test_heapq_cross_check_scores_with_ties():
    """Drained order must match ``heapq`` when the comparison key is the score.

    ``heapq`` orders tuples lexicographically, so ties on the score fall back to
    the doc id and would disagree with our "lower id wins" rule; that is why
    the reference heap here is built on ``(score, 0)`` keys while the doc id
    lives beside it.
    """
    rng = random.Random(7)
    pairs = [(float(rng.randint(0, 4)), rng.randint(0, 20)) for _ in range(200)]
    heap = MinHeap()
    for score, doc_id in pairs:
        heap.push((score, doc_id))
    reference = [(score, 0) for score, _doc_id in pairs]
    heapq.heapify(reference)
    drained = []
    while len(heap) > 0:
        item = heap.pop()
        assert item is not None
        drained.append(item[0])
    reference_drained = []
    while reference:
        reference_drained.append(heapq.heappop(reference)[0])
    assert drained == reference_drained


def test_peek_is_root_and_len_tracks():
    heap = MinHeap()
    assert heap.peek() is None
    assert heap.pop() is None
    heap.push((1.0, 1))
    heap.push((-1.0, 2))
    heap.push((5.0, 3))
    assert heap.peek() == (-1.0, 2)
    assert len(heap) == 3


@pytest.mark.parametrize("k", [1, 3, 10, 50])
def test_top_k_matches_full_sort(k):
    rng = random.Random(k * 17)
    candidates = [(rng.random() * 10, index) for index in range(500)]
    heap = TopKHeap(k)
    for score, doc_id in candidates:
        heap.push(score, doc_id)
    assert heap.get_top_k() == sort_top_k(candidates, k)


def test_tie_break_prefers_lower_doc_id():
    heap = TopKHeap(3)
    for doc_id in (5, 2, 9, 1, 7):
        heap.push(1.0, doc_id)
    assert heap.get_top_k() == [(1.0, 1), (1.0, 2), (1.0, 5)]


def test_top_k_never_exceeds_capacity_and_counts_rejections():
    heap = TopKHeap(4)
    retained = []
    for index in range(100):
        retained.append(heap.push(float(index), index))
    assert len(heap) == 4
    assert sum(retained) == 100  # ascending scores are all retained while filling
    assert heap.get_top_k() == [(99.0, 99), (98.0, 98), (97.0, 97), (96.0, 96)]


def test_top_k_rejects_when_full_and_worse():
    heap = TopKHeap(2)
    assert heap.push(10.0, 1) is True
    assert heap.push(9.0, 2) is True
    assert heap.push(1.0, 3) is False
    assert heap.get_top_k() == [(10.0, 1), (9.0, 2)]


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        TopKHeap(0)
    with pytest.raises(ValueError):
        TopKHeap(-3)


def test_zero_and_negative_scores_are_ranked():
    heap = TopKHeap(3)
    for score, doc_id in [(0.0, 1), (-1.5, 2), (-0.5, 3), (-2.0, 4)]:
        heap.push(score, doc_id)
    assert heap.get_top_k() == [(0.0, 1), (-0.5, 3), (-1.5, 2)]
