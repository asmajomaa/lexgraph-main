"""Clustering quality metrics implemented from scratch.

All three metrics compare a predicted partition with a ground-truth partition:

* **Purity** -- ``(1/n) * sum_c max_j |c AND j|``: the fraction of documents in
  the majority class of their cluster.  Easy to read, biased upward by large
  ``k`` (with ``k = n`` purity is 1).
* **NMI** -- normalised mutual information,
  ``I(C;J) / mean(H(C), H(J))``, using the *arithmetic* mean normalisation
  (the scikit-learn convention) so numbers are comparable to sklearn.
* **ARI** -- the adjusted Rand index, corrected for chance:
  ``(RI - E[RI]) / (max(RI) - E[RI])``.

Nothing here imports numpy or sklearn: the contingency table, the logarithms
and the harmonic/geometric adjustments are all plain Python.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from structures.hash_table import HashSetTable


def contingency_table(
    labels_true: Sequence[str],
    labels_pred: Sequence[int],
) -> HashSetTable:
    """Count documents per ``(true_label, predicted_label)`` cell.

    Args:
        labels_true: Ground-truth labels, one per document.
        labels_pred: Predicted cluster ids, one per document.

    Returns:
        Sparse contingency table ``(true, pred) -> count`` held in a
        :class:`~structures.hash_table.HashSetTable`, so the metrics run on the
        project's own structure rather than a built-in ``dict``.

    Raises:
        ValueError: If the sequences have different lengths.

    Complexity:
        Time O(n); space O(c * j) for c true classes and j clusters actually
        co-occurring.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("label sequences must have the same length")
    table: HashSetTable = HashSetTable()
    for true_label, pred_label in zip(labels_true, labels_pred):
        key = (str(true_label), int(pred_label))
        table.put(key, (table.get(key, 0) or 0) + 1)
    return table


def purity(labels_true: Sequence[str], labels_pred: Sequence[int]) -> float:
    """Cluster purity.

    Args:
        labels_true: Ground-truth labels.
        labels_pred: Predicted cluster ids.

    Returns:
        Value in ``[0, 1]``; 0 for an empty corpus.

    Complexity:
        Time O(n); space O(c * j) for the contingency table.
    """
    n = len(labels_true)
    if n == 0:
        return 0.0
    best_per_cluster: HashSetTable = HashSetTable()
    for (true_label, pred_label), count in contingency_table(labels_true, labels_pred).items():
        del true_label
        if count > (best_per_cluster.get(pred_label, 0) or 0):
            best_per_cluster.put(pred_label, count)
    return sum(best_per_cluster.values()) / n


def _entropy(counts: Sequence[int], total: int) -> float:
    """Shannon entropy in nats of a count distribution.

    Args:
        counts: Non-negative counts.
        total: Their sum.

    Returns:
        ``-sum(p ln p)``; 0 when ``total`` is 0.

    Complexity:
        Time O(len(counts)); space O(1).
    """
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def mutual_information(
    labels_true: Sequence[str], labels_pred: Sequence[int]
) -> float:
    """Mutual information ``I(C; J)`` in nats.

    Args:
        labels_true: Ground-truth labels.
        labels_pred: Predicted cluster ids.

    Returns:
        Non-negative mutual information.

    Complexity:
        Time O(n + c * j); space O(c * j).
    """
    n = len(labels_true)
    if n == 0:
        return 0.0
    table = contingency_table(labels_true, labels_pred)
    true_counts: HashSetTable = HashSetTable()
    pred_counts: HashSetTable = HashSetTable()
    for (true_label, pred_label), count in table.items():
        true_counts.put(true_label, (true_counts.get(true_label, 0) or 0) + count)
        pred_counts.put(pred_label, (pred_counts.get(pred_label, 0) or 0) + count)
    information = 0.0
    for (true_label, pred_label), count in table.items():
        joint = count / n
        information += joint * math.log(
            count * n / (true_counts[true_label] * pred_counts[pred_label])
        )
    return max(information, 0.0)


def nmi(labels_true: Sequence[str], labels_pred: Sequence[int]) -> float:
    """Normalised mutual information with arithmetic-mean normalisation.

    Args:
        labels_true: Ground-truth labels.
        labels_pred: Predicted cluster ids.

    Returns:
        Value in ``[0, 1]``; 1.0 when both partitions are single-class (the
        degenerate case sklearn also reports as 1.0); 0 for an empty corpus.

    Complexity:
        Time O(n + c * j); space O(c * j).
    """
    n = len(labels_true)
    if n == 0:
        return 0.0
    table = contingency_table(labels_true, labels_pred)
    true_counts: HashSetTable = HashSetTable()
    pred_counts: HashSetTable = HashSetTable()
    for (true_label, pred_label), count in table.items():
        true_counts.put(true_label, (true_counts.get(true_label, 0) or 0) + count)
        pred_counts.put(pred_label, (pred_counts.get(pred_label, 0) or 0) + count)
    h_true = _entropy(list(true_counts.values()), n)
    h_pred = _entropy(list(pred_counts.values()), n)
    if h_true == 0.0 and h_pred == 0.0:
        return 1.0
    normaliser = (h_true + h_pred) / 2.0
    if normaliser == 0.0:
        return 0.0
    return mutual_information(labels_true, labels_pred) / normaliser


def _comb2(value: int) -> float:
    """``C(value, 2)`` -- pairs of documents.

    Args:
        value: A count.

    Returns:
        ``value * (value - 1) / 2`` as a float.

    Complexity: O(1).
    """
    return value * (value - 1) / 2.0


def adjusted_rand_index(
    labels_true: Sequence[str], labels_pred: Sequence[int]
) -> float:
    """Adjusted Rand index.

    Computed from the contingency table as

    ``sum_ij C(n_ij, 2)``, ``a = sum_i C(a_i, 2)``, ``b = sum_j C(b_j, 2)``,
    ``expected = a * b / C(n, 2)``, ``maximum = (a + b) / 2``,
    ``ARI = (index - expected) / (maximum - expected)``.

    Args:
        labels_true: Ground-truth labels.
        labels_pred: Predicted cluster ids.

    Returns:
        Value in ``[-0.5, 1]``; 1.0 for identical partitions (including the
        degenerate all-in-one-cluster case); 0.0 for an empty corpus.

    Complexity:
        Time O(n + c * j); space O(c * j).
    """
    n = len(labels_true)
    if n < 2:
        return 1.0 if n == 1 else 0.0
    table = contingency_table(labels_true, labels_pred)
    true_counts: HashSetTable = HashSetTable()
    pred_counts: HashSetTable = HashSetTable()
    cell_pairs = 0.0
    for (true_label, pred_label), count in table.items():
        true_counts.put(true_label, (true_counts.get(true_label, 0) or 0) + count)
        pred_counts.put(pred_label, (pred_counts.get(pred_label, 0) or 0) + count)
        cell_pairs += _comb2(count)
    pairs_true = sum(_comb2(count) for count in true_counts.values())
    pairs_pred = sum(_comb2(count) for count in pred_counts.values())
    total_pairs = _comb2(n)
    if total_pairs == 0.0:
        return 0.0
    expected = pairs_true * pairs_pred / total_pairs
    maximum = (pairs_true + pairs_pred) / 2.0
    if maximum == expected:
        return 1.0
    return (cell_pairs - expected) / (maximum - expected)


def evaluate_clustering(
    labels_true: Sequence[str], labels_pred: Sequence[int]
) -> Dict[str, float]:
    """All three metrics plus basic shape information.

    Args:
        labels_true: Ground-truth labels.
        labels_pred: Predicted cluster ids.

    Returns:
        Mapping with ``purity``, ``nmi``, ``ari``, ``n``, ``n_true``,
        ``n_pred`` and ``largest_cluster_fraction``.

    Complexity:
        Time O(n + c * j) -- the contingency table is built four times by the
        individual metrics, which is still linear and keeps each function
        standalone and testable; space O(c * j).
    """
    clusters: HashSetTable = HashSetTable()
    for label in labels_pred:
        clusters.put(label, (clusters.get(label, 0) or 0) + 1)
    classes: HashSetTable = HashSetTable()
    for label in labels_true:
        key = str(label)
        classes.put(key, (classes.get(key, 0) or 0) + 1)
    largest = max(clusters.values()) / len(labels_pred) if labels_pred else 0.0
    return {
        "purity": purity(labels_true, labels_pred),
        "nmi": nmi(labels_true, labels_pred),
        "ari": adjusted_rand_index(labels_true, labels_pred),
        "n": float(len(labels_true)),
        "n_true": float(len(classes)),
        "n_pred": float(len(clusters)),
        "largest_cluster_fraction": largest,
    }


def confusion_matrix(
    labels_true: Sequence[str], labels_pred: Sequence[int]
) -> Tuple[List[str], List[int], List[List[int]]]:
    """Dense cluster-vs-class confusion matrix for the heatmap figure.

    Args:
        labels_true: Ground-truth labels.
        labels_pred: Predicted cluster ids.

    Returns:
        ``(class_names, cluster_ids, matrix)`` where ``matrix[class][cluster]``
        holds a count.  Both axes are sorted for stable output.

    Complexity:
        Time O(n + c * j); space O(c * j).
    """
    unique_classes: HashSetTable = HashSetTable()
    for label in labels_true:
        unique_classes.put(str(label), 1)
    class_names = sorted(unique_classes.keys())
    unique_clusters: HashSetTable = HashSetTable()
    for label in labels_pred:
        unique_clusters.put(int(label), 1)
    cluster_ids = sorted(unique_clusters.keys())

    class_index: HashSetTable = HashSetTable()
    for index, name in enumerate(class_names):
        class_index.put(name, index)
    cluster_index: HashSetTable = HashSetTable()
    for index, cid in enumerate(cluster_ids):
        cluster_index.put(cid, index)

    matrix = [[0] * len(cluster_ids) for _ in class_names]
    for true_label, pred_label in zip(labels_true, labels_pred):
        matrix[class_index.getitem(str(true_label))][
            cluster_index.getitem(int(pred_label))
        ] += 1
    return class_names, cluster_ids, matrix


__all__ = [
    "contingency_table",
    "purity",
    "nmi",
    "adjusted_rand_index",
    "mutual_information",
    "evaluate_clustering",
    "confusion_matrix",
]
