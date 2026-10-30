"""Binary heap and bounded top-K accumulator, hand-written.

The engine uses a heap in two very different ways:

* :class:`MinHeap` -- a plain array-backed binary heap used for tests,
  Dijkstra-style experiments and the "full sort" correctness oracle.
* :class:`TopKHeap` -- a *bounded* min-heap of capacity K that keeps the K
  largest scores seen so far.  Streaming n candidates through it costs
  ``O(n log K)`` and ``O(K)`` memory instead of ``O(n log n)`` and ``O(n)``
  for sorting the whole candidate list.

Ranking rule: higher score is better; ties are broken by *lower* ``doc_id`` so
that results are deterministic across runs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

#: A ranked item is ``(score, doc_id)``.  Scores are floats; ids are ints.
RankedItem = Tuple[float, int]


def _better(score_a: float, id_a: int, score_b: float, id_b: int) -> bool:
    """Total order on ``(score, doc_id)``: higher score, then lower id.

    Args:
        score_a: First score.
        id_a: First id.
        score_b: Second score.
        id_b: Second id.

    Returns:
        ``True`` when the first item ranks strictly ahead of the second.

    Complexity:
        Time O(1); space O(1).
    """
    if score_a != score_b:
        return score_a > score_b
    return id_a < id_b


class MinHeap:
    """Array-backed binary min-heap ordered by :func:`_better` *reversed*.

    Because "better" means "larger score", a min-heap here keeps the
    *worst-ranked* item at the root, which is exactly what the bounded
    top-K variant needs.

    Attributes:
        _data: The heap array.

    Space: O(n) for n items.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        """Create an empty heap.

        Complexity: O(1).
        """
        self._data: List[RankedItem] = []

    def __len__(self) -> int:
        """Number of items.

        Complexity: O(1).
        """
        return len(self._data)

    def is_empty(self) -> bool:
        """``True`` when the heap holds nothing.

        Complexity: O(1).
        """
        return not self._data

    def _worse(self, a: RankedItem, b: RankedItem) -> bool:
        """``True`` when ``a`` ranks behind ``b`` (a should sit closer to the root).

        Complexity: O(1).
        """
        return _better(b[0], b[1], a[0], a[1])

    def _bubble_up(self, index: int) -> None:
        """Restore heap order by moving ``index`` toward the root.

        Args:
            index: Position of the freshly inserted item.

        Complexity:
            Time O(log n); space O(1).
        """
        data = self._data
        item = data[index]
        while index > 0:
            parent = (index - 1) >> 1
            if not self._worse(item, data[parent]):
                break
            data[index] = data[parent]
            index = parent
        data[index] = item

    def _sink_down(self, index: int) -> None:
        """Restore heap order by pushing ``index`` toward the leaves.

        Args:
            index: Position of the item that may violate the invariant.

        Complexity:
            Time O(log n); space O(1).
        """
        data = self._data
        size = len(data)
        item = data[index]
        child = (index << 1) + 1
        while child < size:
            right = child + 1
            if right < size and self._worse(data[right], data[child]):
                child = right
            if not self._worse(data[child], item):
                break
            data[index] = data[child]
            index = child
            child = (index << 1) + 1
        data[index] = item

    def push(self, item: RankedItem) -> None:
        """Insert one ranked item.

        Args:
            item: ``(score, doc_id)``.

        Complexity:
            Time O(log n); space O(1).
        """
        self._data.append(item)
        self._bubble_up(len(self._data) - 1)

    def pop(self) -> Optional[RankedItem]:
        """Remove and return the worst-ranked item.

        Returns:
            The root item, or ``None`` when empty.

        Complexity:
            Time O(log n); space O(1).
        """
        data = self._data
        if not data:
            return None
        root = data[0]
        last = data.pop()
        if data:
            data[0] = last
            self._sink_down(0)
        return root

    def peek(self) -> Optional[RankedItem]:
        """Return the worst-ranked item without removing it.

        Returns:
            The root item, or ``None`` when empty.

        Complexity:
            Time O(1); space O(1).
        """
        return self._data[0] if self._data else None

    def items(self) -> List[RankedItem]:
        """Copy of the internal array (heap order, not sorted).

        Returns:
            A shallow copy.

        Complexity:
            Time O(n); space O(n).
        """
        return list(self._data)


class TopKHeap:
    """Bounded min-heap retaining the K best ``(score, doc_id)`` items.

    The heap holds at most ``k`` items and its root is the *worst* survivor.
    A new candidate only enters when it beats the root, so streaming n
    candidates costs ``O(n log k)`` time and ``O(k)`` space.

    Attributes:
        k: Capacity.
        _heap: Inner :class:`MinHeap`.

    Space: O(k).
    """

    __slots__ = ("k", "_heap")

    def __init__(self, k: int) -> None:
        """Create a bounded heap.

        Args:
            k: Number of items to retain; must be positive.

        Raises:
            ValueError: If ``k`` is not positive.

        Complexity: Time O(1); space O(k) for the inner heap's capacity.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self._heap = MinHeap()

    def push(self, score: float, doc_id: int) -> bool:
        """Offer one candidate to the top-K set.

        Args:
            score: Relevance score.
            doc_id: Document identifier used for deterministic tie-breaking.

        Returns:
            ``True`` when the candidate was retained.

        Complexity:
            Time O(log k) -- O(1) when the heap is full and the candidate
            loses; space O(1).
        """
        heap = self._heap
        if len(heap) < self.k:
            heap.push((score, doc_id))
            return True
        worst = heap.peek()
        if worst is not None and not _better(score, doc_id, worst[0], worst[1]):
            return False
        heap.pop()
        heap.push((score, doc_id))
        return True

    def get_top_k(self) -> List[RankedItem]:
        """Drain the heap into descending rank order.

        Returns:
            Up to ``k`` ``(score, doc_id)`` pairs, best first.

        Complexity:
            Time O(k log k); space O(k).

        Note:
            Draining mutates the heap, so call this once at the end of a query.
        """
        drained: List[RankedItem] = []
        while not self._heap.is_empty():
            item = self._heap.pop()
            if item is not None:
                drained.append(item)
        drained.reverse()  # min-heap pops worst-first
        return drained

    def __len__(self) -> int:
        """Number of retained items.

        Complexity: O(1).
        """
        return len(self._heap)


def sort_top_k(candidates: List[RankedItem], k: int) -> List[RankedItem]:
    """Reference implementation: full sort, then slice.

    Used as the correctness oracle for :class:`TopKHeap` and by tests.

    Args:
        candidates: ``(score, doc_id)`` pairs.
        k: How many to keep.

    Returns:
        The k best pairs, best first.

    Complexity:
        Time O(n log n); space O(n).
    """
    ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))
    return ordered[:k]


__all__ = ["MinHeap", "TopKHeap", "RankedItem", "sort_top_k"]
