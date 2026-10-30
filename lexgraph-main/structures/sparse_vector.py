"""Sparse vectors as sorted parallel arrays.

A sparse vector of dimension ``d`` stores only the non-zero coordinates, kept
sorted by index so that two vectors can be combined by a linear merge walk
instead of a per-element lookup.  That gives:

* ``O(min(nnz_a, nnz_b))``-ish merge cost for dot products,
* ``O(nnz)`` memory,
* cache-friendly sequential reads.

The representation is deliberately ``(indices: list[int], values: list[float])``
rather than a ``dict`` -- the project's rule forbids dicts inside the index,
and sorted parallel arrays are also faster for the operations K-Means needs.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator, List, Sequence, Tuple


class SparseVector:
    """An immutable-by-convention sparse vector over ``d`` dimensions.

    Attributes:
        dimension: Logical length ``d`` (number of possible coordinates).
        indices: Strictly increasing coordinate indices.
        values: Parallel array of values.

    Space: O(nnz).
    """

    __slots__ = ("dimension", "indices", "values", "_norm")

    def __init__(
        self,
        dimension: int,
        indices: Sequence[int],
        values: Sequence[float],
    ) -> None:
        """Build a sparse vector.

        Args:
            dimension: Logical dimensionality.
            indices: Increasing coordinate indices.
            values: Parallel values.

        Raises:
            ValueError: If lengths differ or indices are not strictly increasing.

        Complexity:
            Time O(nnz) for validation; space O(nnz) for the copies.
        """
        if len(indices) != len(values):
            raise ValueError("indices and values must have the same length")
        previous = -1
        for index in indices:
            if index <= previous:
                raise ValueError("indices must be strictly increasing")
            if index >= dimension:
                raise ValueError(f"index {index} out of range for dimension {dimension}")
            previous = index
        self.dimension = dimension
        self.indices: List[int] = list(indices)
        self.values: List[float] = list(values)
        self._norm: float = -1.0

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_pairs(cls, dimension: int, pairs: Iterable[Tuple[int, float]]) -> "SparseVector":
        """Build from unsorted ``(index, value)`` pairs, summing duplicates.

        Args:
            dimension: Logical dimensionality.
            pairs: Iterable of ``(index, value)``; zero values are dropped.

        Returns:
            A :class:`SparseVector`.

        Complexity:
            Time O(m log m) for m pairs; space O(m).
        """
        merged: List[Tuple[int, float]] = sorted(
            (index, value) for index, value in pairs
        )
        indices: List[int] = []
        values: List[float] = []
        running_index = -1
        running_value = 0.0
        for index, value in merged:
            if value == 0.0:
                continue
            if index == running_index:
                running_value += value
            else:
                if running_index >= 0 and running_value != 0.0:
                    indices.append(running_index)
                    values.append(running_value)
                running_index = index
                running_value = value
        if running_index >= 0 and running_value != 0.0:
            indices.append(running_index)
            values.append(running_value)
        return cls(dimension, indices, values)

    @classmethod
    def zeros(cls, dimension: int) -> "SparseVector":
        """The zero vector of the given dimension.

        Args:
            dimension: Logical dimensionality.

        Returns:
            An empty sparse vector.

        Complexity: O(1).
        """
        return cls(dimension, [], [])

    # -- properties --------------------------------------------------------

    @property
    def nnz(self) -> int:
        """Number of stored coordinates.

        Complexity: O(1).
        """
        return len(self.indices)

    def norm(self) -> float:
        """Euclidean length, cached after the first call.

        Returns:
            ``sqrt(sum(v_i^2))``.

        Complexity:
            Time O(nnz) on first call, O(1) afterwards; space O(1).
        """
        if self._norm < 0.0:
            total = 0.0
            for value in self.values:
                total += value * value
            self._norm = math.sqrt(total)
        return self._norm

    def norm_sq(self) -> float:
        """Squared Euclidean length.

        Returns:
            ``sum(v_i^2)``.

        Complexity:
            Time O(nnz); space O(1).
        """
        total = 0.0
        for value in self.values:
            total += value * value
        return total

    # -- algebra -----------------------------------------------------------

    def dot(self, other: "SparseVector") -> float:
        """Inner product with ``other``.

        Args:
            other: The other sparse vector (same logical dimension).

        Returns:
            ``sum_i a_i * b_i``.

        Complexity:
            Time O(nnz_a + nnz_b) via a merge walk; space O(1).
        """
        a_indices, b_indices = self.indices, other.indices
        a_values, b_values = self.values, other.values
        i = j = 0
        total = 0.0
        na, nb = len(a_indices), len(b_indices)
        while i < na and j < nb:
            a_idx = a_indices[i]
            b_idx = b_indices[j]
            if a_idx == b_idx:
                total += a_values[i] * b_values[j]
                i += 1
                j += 1
            elif a_idx < b_idx:
                i += 1
            else:
                j += 1
        return total

    def dot_dense(self, dense: Sequence[float]) -> float:
        """Inner product with a dense array of length ``dimension``.

        The K-Means centroid is dense (see ``ml/kmeans.py``); pairing it with a
        sparse document through this method keeps the document walk sequential.

        Args:
            dense: Dense vector of length ``self.dimension``.

        Returns:
            ``sum_i v_i * dense[i]``.

        Complexity:
            Time O(nnz); space O(1).  Ideal for sparse document x dense centroid.
        """
        total = 0.0
        for index, value in zip(self.indices, self.values):
            total += value * dense[index]
        return total

    def cosine(self, other: "SparseVector") -> float:
        """Cosine similarity with ``other``.

        Args:
            other: The other sparse vector.

        Returns:
            Value in ``[-1, 1]``; 0 when either vector is zero.

        Complexity:
            Time O(nnz_a + nnz_b); space O(1).
        """
        denominator = self.norm() * other.norm()
        if denominator == 0.0:
            return 0.0
        return self.dot(other) / denominator

    def add(self, other: "SparseVector") -> "SparseVector":
        """Sum with ``other``, produced as a fresh sorted vector.

        Args:
            other: The other sparse vector.

        Returns:
            ``self + other``.

        Complexity:
            Time O(nnz_a + nnz_b); space O(nnz_a + nnz_b).
        """
        a_indices, b_indices = self.indices, other.indices
        a_values, b_values = self.values, other.values
        out_indices: List[int] = []
        out_values: List[float] = []
        i = j = 0
        na, nb = len(a_indices), len(b_indices)
        while i < na or j < nb:
            if j >= nb or (i < na and a_indices[i] < b_indices[j]):
                out_indices.append(a_indices[i])
                out_values.append(a_values[i])
                i += 1
            elif i >= na or b_indices[j] < a_indices[i]:
                out_indices.append(b_indices[j])
                out_values.append(b_values[j])
                j += 1
            else:
                total = a_values[i] + b_values[j]
                if total != 0.0:
                    out_indices.append(a_indices[i])
                    out_values.append(total)
                i += 1
                j += 1
        return SparseVector(self.dimension, out_indices, out_values)

    def scale(self, factor: float) -> "SparseVector":
        """Multiply every coordinate by ``factor``.

        Args:
            factor: Scalar multiplier.

        Returns:
            A new sparse vector.

        Complexity:
            Time O(nnz); space O(nnz).
        """
        return SparseVector(
            self.dimension,
            self.indices,
            [value * factor for value in self.values],
        )

    def squared_distance(self, other: "SparseVector") -> float:
        """Squared Euclidean distance to ``other``.

        For L2-normalised vectors this equals ``2 - 2*cos``, which is why the
        clusterer can use Euclidean K-Means on unit vectors and still behave
        like spherical K-Means.

        Args:
            other: The other sparse vector.

        Returns:
            ``||self - other||^2``.

        Complexity:
            Time O(nnz_a + nnz_b); space O(1).
        """
        a_indices, b_indices = self.indices, other.indices
        a_values, b_values = self.values, other.values
        i = j = 0
        total = 0.0
        na, nb = len(a_indices), len(b_indices)
        while i < na and j < nb:
            a_idx = a_indices[i]
            b_idx = b_indices[j]
            if a_idx == b_idx:
                diff = a_values[i] - b_values[j]
                total += diff * diff
                i += 1
                j += 1
            elif a_idx < b_idx:
                total += a_values[i] * a_values[i]
                i += 1
            else:
                total += b_values[j] * b_values[j]
                j += 1
        while i < na:
            total += a_values[i] * a_values[i]
            i += 1
        while j < nb:
            total += b_values[j] * b_values[j]
            j += 1
        return total

    def to_pairs(self) -> List[Tuple[int, float]]:
        """The ``(index, value)`` pairs as a list.

        Complexity:
            Time O(nnz); space O(nnz).
        """
        return list(zip(self.indices, self.values))

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        """Iterate ``(index, value)`` pairs.

        Complexity: O(nnz).
        """
        return iter(zip(self.indices, self.values))

    def __len__(self) -> int:
        """Number of stored coordinates.

        Complexity: O(1).
        """
        return len(self.indices)

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return f"SparseVector(dim={self.dimension}, nnz={self.nnz})"


__all__ = ["SparseVector"]
