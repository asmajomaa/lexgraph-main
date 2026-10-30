"""Lloyd's K-Means with k-means++ seeding, on sparse TF-IDF vectors.

Representation choice
---------------------
Each centroid is stored **dense** (a ``list[float]`` of length M).  The corpus
is sparse (a document touches tens of the M=5000 columns) but the *centroid of
a cluster with thousands of members* is not: after the first iteration most
columns receive non-zero mass.  Storing centroids densely therefore costs
``O(k * M)`` memory -- 20 * 5000 * 8 bytes = 800 kB -- while making centroid
distance a pure sequential walk.  Sparse centroids would save nothing once
they densify and would add merge overhead on every assignment.

The assignment step is where the sparsity matters: instead of subtracting
dense vectors, we use the identity

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b

so a document's distance to every centroid costs one ``O(nnz)`` sparse-dense
dot product per centroid rather than an ``O(M)`` subtraction per centroid.
With ``nnz ~ 60`` and ``M = 5000`` that is an ~80x reduction in the inner loop.

Stability
---------
Every random draw goes through a local ``random.Random(seed)``.  The global
RNG is never touched, so clustering results are reproducible and do not depend
on whatever else the process is doing.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from config import KMEANS_K, KMEANS_MAX_ITER, KMEANS_PP_TRIALS, KMEANS_SEED, KMEANS_TOL
from structures.sparse_vector import SparseVector


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation.

    Attributes:
        k: Number of clusters.
        max_iter: Hard iteration ceiling.
        tol: Centroid-shift threshold for convergence.
        seed: RNG seed.
        n_trials: Trial points per k-means++ step.
        centroids: Dense centroid vectors (``k`` lists of length M).
        labels: Cluster assignment per training document.
        inertia: Sum of squared distances to the assigned centroid.
        inertia_history: Inertia after each iteration, for the report figure.
        shift_history: Maximum centroid shift per iteration.
        n_iter: Iterations actually run.
        converged: Why fitting stopped (``"converged"`` or ``"max_iter"``).

    Space: O(k * M) for centroids plus O(n) for labels.
    """

    __slots__ = (
        "k",
        "max_iter",
        "tol",
        "seed",
        "n_trials",
        "centroids",
        "labels",
        "inertia",
        "inertia_history",
        "shift_history",
        "n_iter",
        "converged",
        "_rng",
        "_centroid_norms",
    )

    def __init__(
        self,
        k: int = KMEANS_K,
        max_iter: int = KMEANS_MAX_ITER,
        tol: float = KMEANS_TOL,
        seed: int = KMEANS_SEED,
        n_trials: int = KMEANS_PP_TRIALS,
    ) -> None:
        """Configure the clusterer.

        Args:
            k: Number of clusters; must be positive.
            max_iter: Iteration ceiling.
            tol: Convergence threshold on the maximum centroid L2 shift.
            seed: RNG seed for k-means++ and empty-cluster reseeding.
            n_trials: Trial points sampled per k-means++ step.

        Complexity: O(1).
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.n_trials = n_trials
        self.centroids: List[List[float]] = []
        self.labels: List[int] = []
        self.inertia = 0.0
        self.inertia_history: List[float] = []
        self.shift_history: List[float] = []
        self.n_iter = 0
        self.converged = ""
        self._rng = random.Random(seed)
        self._centroid_norms: List[float] = []

    # -- helpers -----------------------------------------------------------

    def _sq_distance(self, vector: SparseVector, centroid: List[float], norm_sq: float) -> float:
        """Squared Euclidean distance between a sparse vector and a dense centroid.

        Args:
            vector: Unit-norm or arbitrary sparse vector.
            centroid: Dense centroid.
            norm_sq: ``||centroid||^2``, precomputed for the caller.

        Returns:
            Squared distance, clamped at 0 against float round-off.

        Complexity:
            Time O(nnz); space O(1).
        """
        dot = 0.0
        for index, value in zip(vector.indices, vector.values):
            dot += value * centroid[index]
        distance = norm_sq + vector.norm_sq() - 2.0 * dot
        return distance if distance > 0.0 else 0.0

    def _closest(
        self, vector: SparseVector, vector_norm_sq: float
    ) -> Tuple[int, float]:
        """Index and squared distance of the nearest centroid.

        Args:
            vector: The point.
            vector_norm_sq: ``||vector||^2``, precomputed.

        Returns:
            ``(cluster_index, squared_distance)``.

        Complexity:
            Time O(k * nnz); space O(1).
        """
        best_index = 0
        best_distance = float("inf")
        for cluster in range(len(self.centroids)):
            centroid = self.centroids[cluster]
            norm_sq = self._centroid_norms[cluster]
            dot = 0.0
            for index, value in zip(vector.indices, vector.values):
                dot += value * centroid[index]
            distance = norm_sq + vector_norm_sq - 2.0 * dot
            if distance < best_distance:
                best_distance = distance
                best_index = cluster
        return best_index, best_distance if best_distance > 0.0 else 0.0

    # -- initialisation ----------------------------------------------------

    def _kmeanspp(self, vectors: Sequence[SparseVector]) -> List[List[float]]:
        """Choose ``k`` starting centroids with k-means++ sampling.

        The first centre is uniform; each subsequent centre is drawn from the
        remaining points with probability proportional to ``D(x)^2`` (squared
        distance to the nearest chosen centre).  ``n_trials`` samples are drawn
        per step and the best is kept, which approximates the greedy variant
        with a constant-factor quality guarantee at ``O(log k)`` extra draws.

        Args:
            vectors: Training vectors.

        Returns:
            ``k`` dense centroid vectors (copies).

        Complexity:
            Time O(k * n_trials * n * nnz) -- each step scores every point
            against the chosen set, which is the textbook cost; space O(k * M).
        """
        n = len(vectors)
        dimension = vectors[0].dimension if n else 0
        chosen: List[int] = [self._rng.randrange(n)]

        # distances[i] = squared distance from point i to its nearest chosen centre
        first = vectors[chosen[0]]
        distances = [self._sq_distance_to_sparse(vectors[i], first) for i in range(n)]

        while len(chosen) < self.k and len(chosen) < n:
            best_index = -1
            best_potential = float("inf")
            for _ in range(self.n_trials):
                index = self._sample_weighted(distances)
                if index < 0:
                    break
                candidate = vectors[index]
                potential = 0.0
                for i in range(n):
                    d = self._sq_distance_to_sparse(vectors[i], candidate)
                    potential += d if d < distances[i] else distances[i]
                if potential < best_potential:
                    best_potential = potential
                    best_index = index
            if best_index < 0:
                break
            chosen.append(best_index)
            candidate = vectors[best_index]
            for i in range(n):
                d = self._sq_distance_to_sparse(vectors[i], candidate)
                if d < distances[i]:
                    distances[i] = d

        while len(chosen) < self.k:
            chosen.append(self._rng.randrange(n))

        return [list(self._to_dense(vectors[index], dimension)) for index in chosen]

    def _sq_distance_to_sparse(self, a: SparseVector, b: SparseVector) -> float:
        """Squared distance between two sparse vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Squared Euclidean distance.

        Complexity:
            Time O(nnz_a + nnz_b); space O(1).
        """
        return a.squared_distance(b)

    def _sample_weighted(self, weights: Sequence[float]) -> int:
        """Pick an index with probability proportional to ``weights[i]``.

        Args:
            weights: Non-negative weights.

        Returns:
            Chosen index, or ``-1`` when all weights are zero.

        Complexity:
            Time O(n); space O(1).
        """
        total = 0.0
        for weight in weights:
            if weight > 0.0:
                total += weight
        if total <= 0.0:
            return -1
        target = self._rng.random() * total
        running = 0.0
        last_positive = -1
        for index, weight in enumerate(weights):
            if weight <= 0.0:
                continue
            last_positive = index
            running += weight
            if running >= target:
                return index
        return last_positive

    @staticmethod
    def _to_dense(vector: SparseVector, dimension: int) -> List[float]:
        """Expand a sparse vector into a dense list.

        Args:
            vector: Sparse vector.
            dimension: Output length.

        Returns:
            Dense list.

        Complexity:
            Time O(nnz + M); space O(M).
        """
        dense = [0.0] * dimension
        for index, value in zip(vector.indices, vector.values):
            dense[index] = value
        return dense

    # -- fitting -----------------------------------------------------------

    def fit(self, vectors: Sequence[SparseVector]) -> "KMeans":
        """Run Lloyd's algorithm to convergence.

        Args:
            vectors: Training vectors, all of the same dimension.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``vectors`` is empty.

        Complexity:
            Time O(n * k * nnz * iterations) using the sparse-dense distance
            identity; the naive dense form would be O(n * k * M * iterations).
            Space O(k * M + n) for centroids, accumulator and labels.
        """
        n = len(vectors)
        if n == 0:
            raise ValueError("cannot fit on an empty corpus")
        dimension = vectors[0].dimension
        k = min(self.k, n)

        self.centroids = self._kmeanspp(vectors)
        self._centroid_norms = [
            sum(value * value for value in centroid) for centroid in self.centroids
        ]
        norms_sq = [vector.norm_sq() for vector in vectors]
        labels = [-1] * n
        self.inertia_history = []
        self.shift_history = []

        for iteration in range(self.max_iter):
            changed = 0
            inertia = 0.0
            for index in range(n):
                cluster, distance = self._closest(vectors[index], norms_sq[index])
                if cluster != labels[index]:
                    labels[index] = cluster
                    changed += 1
                inertia += distance
            self.inertia_history.append(inertia)

            # Accumulate sums sparsely: each document adds its few non-zero
            # coordinates to its cluster's accumulator instead of walking M.
            sums = [[0.0] * dimension for _ in range(k)]
            counts = [0] * k
            for index in range(n):
                cluster = labels[index]
                counts[cluster] += 1
                accumulator = sums[cluster]
                for coordinate, value in zip(vectors[index].indices, vectors[index].values):
                    accumulator[coordinate] += value

            max_shift = 0.0
            for cluster in range(k):
                if counts[cluster] == 0:
                    # Empty cluster: reseed to the farthest point from its own
                    # centroid, which keeps k clusters alive without inventing
                    # a random direction on a normalised sphere.
                    farthest_index, _distance = self._farthest(vectors, labels, norms_sq)
                    sums[cluster] = self._to_dense(vectors[farthest_index], dimension)
                    counts[cluster] = 1
                inverse = 1.0 / counts[cluster]
                centroid = self.centroids[cluster]
                shift = 0.0
                for coordinate in range(dimension):
                    updated = sums[cluster][coordinate] * inverse
                    delta = updated - centroid[coordinate]
                    shift += delta * delta
                    centroid[coordinate] = updated
                shift = math.sqrt(shift)
                if shift > max_shift:
                    max_shift = shift

            self._centroid_norms = [
                sum(value * value for value in centroid) for centroid in self.centroids
            ]
            self.shift_history.append(max_shift)
            self.n_iter = iteration + 1
            if changed == 0 or max_shift < self.tol:
                self.converged = "converged"
                break
        else:
            self.converged = "max_iter"

        self.labels = labels
        self.inertia = self.inertia_history[-1] if self.inertia_history else 0.0
        return self

    def _farthest(
        self,
        vectors: Sequence[SparseVector],
        labels: Sequence[int],
        norms_sq: Sequence[float],
    ) -> Tuple[int, float]:
        """Find the point with the largest distance to its assigned centroid.

        Args:
            vectors: Training vectors.
            labels: Current assignments.
            norms_sq: Precomputed squared norms.

        Returns:
            ``(index, distance)``; ``(0, 0.0)`` for an empty corpus.

        Complexity:
            Time O(n * nnz); space O(1).
        """
        best_index = 0
        best_distance = -1.0
        for index in range(len(vectors)):
            centroid = self.centroids[labels[index]]
            distance = self._sq_distance(
                vectors[index], centroid, self._centroid_norms[labels[index]]
            )
            if distance > best_distance:
                best_distance = distance
                best_index = index
        return best_index, max(best_distance, 0.0)

    # -- inference ---------------------------------------------------------

    def predict(self, vector: SparseVector) -> int:
        """Nearest centroid for one vector.

        Args:
            vector: Query vector.

        Returns:
            Cluster index.

        Complexity:
            Time O(k * nnz); space O(1).
        """
        return self._closest(vector, vector.norm_sq())[0]

    def predict_top_n(self, vector: SparseVector, n: int = 2) -> List[Tuple[int, float]]:
        """The ``n`` nearest centroids, nearest first.

        Args:
            vector: Query vector.
            n: How many clusters to return.

        Returns:
            ``(cluster_index, squared_distance)`` pairs.

        Complexity:
            Time O(k * nnz + k log k); space O(k).
        """
        if not self.centroids:
            return []
        vector_norm_sq = vector.norm_sq()
        scored: List[Tuple[int, float]] = []
        for cluster in range(len(self.centroids)):
            scored.append(
                (
                    cluster,
                    self._sq_distance(
                        vector,
                        self.centroids[cluster],
                        self._centroid_norms[cluster],
                    ),
                )
            )
        scored.sort(key=lambda pair: (pair[1], pair[0]))
        return scored[: max(1, n)]

    def predict_batch(self, vectors: Sequence[SparseVector]) -> List[int]:
        """Assignments for many vectors.

        Args:
            vectors: Query vectors.

        Returns:
            Cluster index per vector.

        Complexity:
            Time O(n * k * nnz); space O(n).
        """
        return [self.predict(vector) for vector in vectors]

    def cluster_members(self, labels: Sequence[int]) -> List[List[int]]:
        """Group document indices by cluster.

        Args:
            labels: One label per document.

        Returns:
            ``k`` lists of document indices.

        Complexity:
            Time O(n); space O(n).
        """
        members: List[List[int]] = [[] for _ in range(len(self.centroids))]
        for index, label in enumerate(labels):
            if 0 <= label < len(members):
                members[label].append(index)
        return members

    def to_state(self) -> Dict[str, object]:
        """Serialise the fitted model.

        Returns:
            A picklable mapping.

        Complexity:
            Time O(k * M); space O(k * M).
        """
        return {
            "k": self.k,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "centroids": self.centroids,
            "labels": self.labels,
            "inertia": self.inertia,
            "inertia_history": self.inertia_history,
            "shift_history": self.shift_history,
            "n_iter": self.n_iter,
            "converged": self.converged,
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "KMeans":
        """Rebuild a fitted model from :meth:`to_state` output.

        Args:
            state: Output of :meth:`to_state`.

        Returns:
            A fitted :class:`KMeans`.

        Complexity:
            Time O(k * M); space O(k * M).
        """
        model = cls(
            k=int(state["k"]),  # type: ignore[arg-type]
            max_iter=int(state["max_iter"]),  # type: ignore[arg-type]
            tol=float(state["tol"]),  # type: ignore[arg-type]
            seed=int(state["seed"]),  # type: ignore[arg-type]
            n_trials=int(state["n_trials"]),  # type: ignore[arg-type]
        )
        model.centroids = [  # type: ignore[arg-type]
            list(centroid) for centroid in state["centroids"]
        ]
        model.labels = list(state["labels"])  # type: ignore[arg-type]
        model.inertia = float(state["inertia"])  # type: ignore[arg-type]
        model.inertia_history = list(state["inertia_history"])  # type: ignore[arg-type]
        model.shift_history = list(state["shift_history"])  # type: ignore[arg-type]
        model.n_iter = int(state["n_iter"])  # type: ignore[arg-type]
        model.converged = str(state["converged"])
        model._centroid_norms = [
            sum(value * value for value in centroid) for centroid in model.centroids
        ]
        return model

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return (
            f"KMeans(k={self.k}, iters={self.n_iter}, "
            f"inertia={self.inertia:.4f}, {self.converged})"
        )


__all__ = ["KMeans"]
