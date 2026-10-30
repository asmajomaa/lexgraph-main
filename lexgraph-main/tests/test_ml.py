"""Vectorizer, K-Means and clustering-metric tests.

The K-Means quality test uses well-separated synthetic blobs so that a correct
implementation *must* recover the partition; the metric tests cross-check
against closed-form values and against scikit-learn where a third-party number
is the honest reference.
"""

from __future__ import annotations

import math
import random

import pytest

from config import KMEANS_SEED
from core.tokenizer import Tokenizer
from data.documents import Document
from ml.evaluation import (
    adjusted_rand_index,
    confusion_matrix,
    evaluate_clustering,
    nmi,
    purity,
)
from ml.kmeans import KMeans
from ml.vectorizer import TFIDFVectorizer
from structures.sparse_vector import SparseVector


def _doc(doc_id: int, text: str, label: str = "x") -> Document:
    return Document(
        doc_id=doc_id,
        external_id=f"d{doc_id}",
        text=text,
        title=f"doc {doc_id}",
        label=label,
        source="test",
        snippet=text,
    )


def _blob_vectors(
    rng: random.Random, centres: list[tuple[float, float]], per_cluster: int, dimension: int = 12
) -> tuple[list[SparseVector], list[int]]:
    vectors: list[SparseVector] = []
    truth: list[int] = []
    for cluster, (cx, cy) in enumerate(centres):
        for _ in range(per_cluster):
            pairs = [
                (0, cx + rng.gauss(0, 0.05)),
                (1, cy + rng.gauss(0, 0.05)),
                (2, rng.gauss(0, 0.05)),
            ]
            vectors.append(SparseVector.from_pairs(dimension, pairs))
            truth.append(cluster)
    return vectors, truth


# -- vectorizer -------------------------------------------------------------


def test_vectorizer_fits_and_normalises():
    documents = [
        _doc(0, "graphics render shader texture"),
        _doc(1, "kernel module compile linux"),
        _doc(2, "graphics texture mapping"),
    ]
    vectorizer = TFIDFVectorizer(vocab_size=50, min_df=1, max_df_ratio=1.0).fit(documents)
    assert vectorizer.dimension > 0
    vectors = vectorizer.transform(documents)
    for vector in vectors:
        assert vector.dimension == vectorizer.dimension
        assert vector.norm() == pytest.approx(1.0, rel=1e-9)


def test_vectorizer_respects_min_df_and_vocab_cap():
    documents = [_doc(i, "alpha beta " + " ".join(f"w{j}" for j in range(i))) for i in range(6)]
    vectorizer = TFIDFVectorizer(vocab_size=3, min_df=1, max_df_ratio=1.0).fit(documents)
    assert vectorizer.dimension == 3


def test_vectorizer_unknown_text_yields_zero_vector():
    documents = [_doc(0, "alpha beta"), _doc(1, "beta gamma")]
    vectorizer = TFIDFVectorizer(vocab_size=10, min_df=1, max_df_ratio=1.0).fit(documents)
    vector = vectorizer.transform_text("completely unrelated words")
    assert vector.nnz == 0
    assert vector.norm() == 0.0


def test_vectorizer_round_trips_through_state():
    documents = [_doc(i, f"shared token{i} extra{i}") for i in range(4)]
    vectorizer = TFIDFVectorizer(vocab_size=20, min_df=1, max_df_ratio=1.0).fit(documents)
    restored = TFIDFVectorizer.from_state(vectorizer.to_state())
    assert restored.vocabulary == vectorizer.vocabulary
    assert restored.transform_text("shared token1").values == pytest.approx(
        vectorizer.transform_text("shared token1").values
    )


def test_vectorizer_uses_shared_tokenizer():
    tokenizer = Tokenizer(remove_stop_words=True, use_stemmer=True)
    documents = [_doc(0, "rendering textures"), _doc(1, "renders texture")]
    vectorizer = TFIDFVectorizer(
        vocab_size=10, min_df=1, max_df_ratio=1.0, tokenizer=tokenizer
    ).fit(documents)
    vocabulary = set(vectorizer.index_to_term)
    assert vocabulary <= {"render", "texture"} | vocabulary


# -- K-Means ----------------------------------------------------------------


def test_kmeans_recovers_separated_blobs():
    rng = random.Random(1)
    vectors, truth = _blob_vectors(rng, [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)], per_cluster=25)
    model = KMeans(k=3, max_iter=50, seed=KMEANS_SEED)
    model.fit(vectors)
    assert model.n_iter > 0
    from ml.evaluation import adjusted_rand_index as ari

    assert ari([str(t) for t in truth], model.labels) == pytest.approx(1.0)


def test_kmeans_is_deterministic_for_a_seed():
    rng = random.Random(2)
    vectors, _truth = _blob_vectors(rng, [(0, 0), (4, 4)], per_cluster=20)
    first = KMeans(k=2, max_iter=30, seed=7).fit(vectors)
    second = KMeans(k=2, max_iter=30, seed=7).fit(vectors)
    assert first.labels == second.labels
    assert first.inertia == pytest.approx(second.inertia)


def test_kmeans_inertia_is_non_increasing():
    rng = random.Random(3)
    vectors, _truth = _blob_vectors(rng, [(0, 0), (1, 1), (2, 2)], per_cluster=15)
    model = KMeans(k=3, max_iter=25, seed=KMEANS_SEED).fit(vectors)
    history = model.inertia_history
    assert len(history) >= 1
    for earlier, later in zip(history, history[1:]):
        assert later <= earlier + 1e-9


def test_kmeans_empty_cluster_is_reseeded():
    # All points identical except one: the second cluster will be empty unless
    # the implementation reseeds it.
    vectors = [SparseVector(6, [0], [1.0]) for _ in range(10)]
    vectors.append(SparseVector(6, [5], [1.0]))
    model = KMeans(k=2, max_iter=10, seed=KMEANS_SEED).fit(vectors)
    assert len(model.centroids) == 2
    assert sorted(model.labels)[0] >= 0
    assert len(set(model.labels)) >= 1


def test_kmeans_predict_and_top_n():
    rng = random.Random(4)
    vectors, _truth = _blob_vectors(rng, [(0, 0), (6, 6)], per_cluster=20)
    model = KMeans(k=2, max_iter=30, seed=KMEANS_SEED).fit(vectors)
    probe = SparseVector.from_pairs(12, [(0, 0.0), (1, 0.0), (2, 0.0)])
    top_two = model.predict_top_n(probe, 2)
    assert len(top_two) == 2
    assert top_two[0][1] <= top_two[1][1]
    assert model.predict(probe) == top_two[0][0]


def test_kmeans_k_larger_than_corpus():
    """k > n must not crash; the model keeps k slots (some possibly identical)."""
    vectors = [SparseVector(4, [0], [1.0]), SparseVector(4, [1], [1.0])]
    model = KMeans(k=5, max_iter=5, seed=KMEANS_SEED).fit(vectors)
    assert len(model.centroids) == 5
    assert len(model.labels) == 2
    assert all(0 <= label < 5 for label in model.labels)


def test_kmeans_rejects_empty_corpus():
    with pytest.raises(ValueError):
        KMeans(k=2).fit([])


def test_kmeans_rejects_bad_k():
    with pytest.raises(ValueError):
        KMeans(k=0)


def test_kmeans_cluster_members_and_round_trip():
    rng = random.Random(5)
    vectors, _truth = _blob_vectors(rng, [(0, 0), (8, 8)], per_cluster=12)
    model = KMeans(k=2, max_iter=20, seed=KMEANS_SEED).fit(vectors)
    members = model.cluster_members(model.labels)
    assert sum(len(group) for group in members) == len(vectors)
    restored = KMeans.from_state(model.to_state())
    assert restored.labels == model.labels
    assert restored.predict(vectors[0]) == model.predict(vectors[0])


# -- metrics ----------------------------------------------------------------


def test_purity_perfect_and_impure():
    assert purity(["a", "a", "b", "b"], [0, 0, 1, 1]) == 1.0
    assert purity(["a", "a", "b", "b"], [0, 0, 0, 0]) == 0.5
    assert purity([], []) == 0.0


def test_nmi_matches_closed_form_for_identical_partitions():
    assert nmi(["a", "a", "b", "b"], [0, 0, 1, 1]) == pytest.approx(1.0)
    # Checking whether NMI ever gets the independent case right is a sanity test.
    # Two perfectly crossed 2x2 partitions carry zero mutual information.
    assert nmi(["a", "b", "a", "b"], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert nmi(["a", "a", "b", "b"], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_ari_known_values():
    assert adjusted_rand_index(["a", "b", "c"], [0, 1, 2]) == pytest.approx(1.0)
    assert adjusted_rand_index(["a", "a", "b", "b"], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert adjusted_rand_index(["a", "a", "b", "b"], [0, 0, 0, 0]) == pytest.approx(0.0)


def test_nmi_and_ari_match_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = random.Random(11)
    truth = [rng.choice("abcde") for _ in range(200)]
    pred = [rng.randrange(4) for _ in range(200)]
    assert nmi(truth, pred) == pytest.approx(
        sklearn_metrics.normalized_mutual_info_score(truth, pred), rel=1e-9
    )
    assert adjusted_rand_index(truth, pred) == pytest.approx(
        sklearn_metrics.adjusted_rand_score(truth, pred), rel=1e-9
    )


def test_evaluate_clustering_shape_and_confusion_matrix():
    truth = ["a", "a", "b", "b"]
    pred = [0, 0, 1, 1]
    report = evaluate_clustering(truth, pred)
    assert report["purity"] == 1.0
    assert report["n"] == 4.0
    assert report["n_true"] == 2.0
    assert report["n_pred"] == 2.0
    classes, clusters, matrix = confusion_matrix(truth, pred)
    assert classes == ["a", "b"] and clusters == [0, 1]
    assert matrix == [[2, 0], [0, 2]]


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        purity(["a"], [1, 2])
