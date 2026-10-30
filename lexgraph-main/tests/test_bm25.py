"""Inverted index and BM25 tests, including a hand-computed three-document case.

Worked example (also used verbatim in the report)
-------------------------------------------------
Corpus of three documents, tokenised with stop words removed::

    d0: "graphics render texture"   -> 3 tokens
    d1: "graphics render shader"    -> 3 tokens
    d2: "kernel boot linux"         -> 3 tokens

Query: ``"graphics render"`` with ``k1 = 1.5``, ``b = 0.75``.

``N = 3``, ``avgdl = 3``, so the length factor ``1 - b + b*dl/avgdl = 1`` for
every document and the denominator reduces to ``tf + 1.5``.

* ``n(graphics) = 2``:
  ``idf = ln(1 + (3 - 2 + 0.5)/(2 + 0.5)) = ln(1.6) = 0.4700036292``
* ``n(render) = 2``: same idf.
* ``d0``: tf=1 for both terms ->
  ``2 * 0.4700036292 * 1 * 2.5 / (1 + 1.5) = 0.9400072585``
* ``d1``: identical tf counts -> same score.
* ``d2``: no matching term -> 0.

Since scores tie, the deterministic tie-break by lower ``doc_id`` puts ``d0``
first.
"""

from __future__ import annotations

import math

import pytest

from core.bm25 import BM25
from core.inverted_index import CandidateSet, InvertedIndex, RestrictedView
from core.tokenizer import Tokenizer
from data.documents import Document


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


@pytest.fixture()
def tiny_index() -> InvertedIndex:
    tokenizer = Tokenizer(remove_stop_words=True, use_stemmer=False)
    documents = [
        _doc(0, "graphics render texture"),
        _doc(1, "graphics render shader"),
        _doc(2, "kernel boot linux"),
    ]
    return InvertedIndex.build(documents, tokenizer, name="tiny")


def test_index_statistics(tiny_index):
    assert tiny_index.N == 3
    assert tiny_index.average_document_length == pytest.approx(3.0)
    assert tiny_index.doc_lengths == [3, 3, 3]
    assert tiny_index.total_length == 9
    assert tiny_index.document_frequency("graphics") == 2
    assert tiny_index.document_frequency("missing") == 0
    assert tiny_index.postings_for("graphics") == [(0, 1), (1, 1)]
    assert tiny_index.vocabulary_size() == 7


def test_index_rejects_non_dense_ids():
    index = InvertedIndex()
    with pytest.raises(ValueError):
        index.add_document(_doc(5, "gap"))


def test_bm25_hand_computed_three_documents(tiny_index):
    scorer = BM25(tiny_index, k1=1.5, b=0.75)
    idf = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = 2 * idf * (1 * 2.5) / (1 + 1.5)
    results = scorer.search("graphics render", k=3)
    assert [doc_id for _score, doc_id in results] == [0, 1]
    assert results[0][0] == pytest.approx(expected, rel=1e-12)
    assert results[1][0] == pytest.approx(expected, rel=1e-12)
    assert scorer.score_document("graphics render", 2) == 0.0


def test_bm25_explain_matches_score(tiny_index):
    scorer = BM25(tiny_index)
    rows = scorer.explain("graphics render", 0)
    total = sum(row[3] for row in rows)
    assert total == pytest.approx(scorer.score_document("graphics render", 0))
    assert [row[0] for row in rows] == ["graphics", "render"]
    text = scorer.explain_text("graphics render", 0)
    assert "total =" in text and "avgdl" in text


def test_idf_decreases_with_document_frequency():
    scorer = BM25(InvertedIndex())
    # N=100: n=1 -> ln(1+99.5/1.5), n=99 -> ln(1+1.5/99.5).  Strictly decreasing
    # and never zero, because the "+1" inside the log keeps the ratio above 1.
    assert scorer.idf(1, 100) > scorer.idf(10, 100) > scorer.idf(50, 100) > 0.0
    assert scorer.idf(100, 100) > 0.0
    assert scorer.idf(0, 100) == 0.0
    assert scorer.idf(5, 0) == 0.0


def test_length_normalisation_discounts_long_documents():
    tokenizer = Tokenizer(remove_stop_words=False)
    long_text = "graphics " + " ".join(f"w{i}" for i in range(50))
    documents = [_doc(0, "graphics"), _doc(1, long_text), _doc(2, "other stuff")]
    index = InvertedIndex.build(documents, tokenizer)
    scorer = BM25(index, k1=1.5, b=0.75)
    short_score = scorer.score_document("graphics", 0)
    long_score = scorer.score_document("graphics", 1)
    assert short_score > long_score > 0.0


def test_b_parameter_zero_disables_length_normalisation():
    tokenizer = Tokenizer(remove_stop_words=False)
    documents = [_doc(0, "graphics"), _doc(1, "graphics " + " ".join(f"w{i}" for i in range(50)))]
    index = InvertedIndex.build(documents, tokenizer)
    scorer = BM25(index, b=0.0)
    assert scorer.score_document("graphics", 0) == pytest.approx(
        scorer.score_document("graphics", 1)
    )


def test_empty_query_and_unknown_terms(tiny_index):
    scorer = BM25(tiny_index)
    assert scorer.search("", k=5) == []
    assert scorer.search("zzzz", k=5) == []
    assert scorer.search("the of and", k=5) == []
    assert scorer.scored_documents == 0


def test_k_larger_than_results(tiny_index):
    scorer = BM25(tiny_index)
    assert len(scorer.search("graphics", k=50)) == 2


def test_top_k_ordering_and_limit(tiny_index):
    scorer = BM25(tiny_index)
    results = scorer.search("graphics render", k=1)
    assert len(results) == 1
    assert results[0][1] == 0


def test_restricted_view_filters_candidates(tiny_index):
    view = RestrictedView(tiny_index, CandidateSet([1]))
    scorer = BM25(view)
    results = scorer.search("graphics render", k=5)
    assert [doc_id for _score, doc_id in results] == [1]
    # The restricted view reports the *global* df on purpose (see its
    # docstring): re-weighting idf per candidate set would make the pruned and
    # baseline rankings incomparable.
    assert view.document_frequency("graphics") == 2
    stats = view.collection_statistics()
    assert stats["N"] == 1.0 and stats["corpus_N"] == 3.0


def test_cluster_filter_membership():
    from core.inverted_index import ClusterFilter

    cluster_of = [0, 0, 1, 2, 2, 1]
    filt = ClusterFilter(cluster_of, [1], n_clusters=3)
    assert filt.contains(2) and filt.contains(5)
    assert not filt.contains(0) and not filt.contains(3)
    assert not filt.contains(-1) and not filt.contains(99)
    assert filt.pool_size() == 2


def test_candidate_set_membership_and_ids():
    candidates = CandidateSet([3, 1, 2])
    assert candidates.contains(2) and not candidates.contains(9)
    assert candidates.ids() == [1, 2, 3]
    assert len(candidates) == 3


def test_index_round_trips_through_state(tiny_index):
    restored = InvertedIndex.from_state(tiny_index.to_state())
    assert restored.N == tiny_index.N
    assert restored.doc_lengths == tiny_index.doc_lengths
    assert restored.titles == tiny_index.titles
    assert BM25(restored).search("graphics render", 3) == BM25(tiny_index).search(
        "graphics render", 3
    )


def test_stats_reporting_after_search(tiny_index):
    scorer = BM25(tiny_index)
    scorer.search("graphics", 5)
    assert scorer.last_stats["terms"] == 1.0
    assert scorer.scored_documents == 2
