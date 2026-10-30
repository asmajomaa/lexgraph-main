"""Okapi BM25 scoring on top of the custom inverted index and heap.

Formula (as specified)::

    idf(t)  = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(q, d) = sum_t idf(t) * tf * (k1 + 1)
                  / (tf + k1 * (1 - b + b * dl / avgdl))

with ``k1 = 1.5`` and ``b = 0.75`` by default.

Two evaluation strategies are implemented:

* **Term-at-a-time** (default, :meth:`BM25.search`): walk the query terms, and
  for each term walk its postings, accumulating into a score array.  This is
  the standard inverted-index strategy and it touches each posting once.
* **Document-at-a-time** (:meth:`BM25.score_document`): score one document
  from scratch.  Slower for search but exact and easy to hand-trace, so it
  backs ``explain`` and the report's worked example.

Both feed results through :class:`structures.min_heap.TopKHeap`, giving
``O(P log K)`` where P is the number of postings visited.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from config import BM25_B, BM25_K1
from core.inverted_index import InvertedIndex, RestrictedView
from structures.hash_table import HashSetTable
from structures.min_heap import RankedItem, TopKHeap


class BM25:
    """BM25 scorer bound to one index (full or candidate-restricted).

    Attributes:
        index: The :class:`InvertedIndex` or :class:`RestrictedView`.
        k1: Term-frequency saturation.
        b: Length-normalisation strength.
        scored_documents: Number of posting visits in the last search, for
            benchmarking.

    Space: O(1) beyond the referenced index.
    """

    __slots__ = ("index", "k1", "b", "scored_documents", "last_stats")

    def __init__(
        self,
        index: object,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Bind the scorer.

        Args:
            index: :class:`InvertedIndex` or :class:`RestrictedView`.
            k1: Term-frequency saturation parameter.
            b: Length-normalisation parameter.

        Complexity: O(1).
        """
        self.index = index
        self.k1 = k1
        self.b = b
        self.scored_documents = 0
        self.last_stats: Dict[str, float] = {}

    # -- idf ---------------------------------------------------------------

    def idf(self, document_frequency: int, n_docs: int) -> float:
        """Inverse document frequency.

        Args:
            document_frequency: ``n_t`` -- number of documents containing the term.
            n_docs: ``N`` -- corpus size.

        Returns:
            ``ln(1 + (N - n + 0.5) / (n + 0.5))``; 0 when the term is unknown or
            the corpus is empty.  The ``+1`` inside the logarithm keeps the
            numerator above the denominator, so idf stays strictly positive even
            for a term present in every document -- this is the standard Lucene
            variant and is what makes single-term corpora scoreable.

        Complexity:
            Time O(1); space O(1).
        """
        if document_frequency <= 0 or n_docs <= 0:
            return 0.0
        ratio = 1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5)
        if ratio <= 1.0:
            return 0.0
        return math.log(ratio)

    # -- single term -------------------------------------------------------

    def term_score(
        self,
        term_frequency: int,
        document_length: int,
        average_length: float,
        idf: float,
    ) -> float:
        """Contribution of one term to one document.

        Args:
            term_frequency: ``tf`` in the document.
            document_length: ``dl`` in tokens.
            average_length: ``avgdl`` of the corpus.
            idf: Precomputed idf for the term.

        Returns:
            The BM25 term contribution.

        Complexity:
            Time O(1); space O(1).
        """
        if term_frequency <= 0 or idf == 0.0:
            return 0.0
        if average_length <= 0.0:
            average_length = 1.0
        numerator = term_frequency * (self.k1 + 1.0)
        denominator = term_frequency + self.k1 * (
            1.0 - self.b + self.b * document_length / average_length
        )
        return idf * numerator / denominator

    # -- query handling ----------------------------------------------------

    def _prepare_query(self, query: str) -> Tuple[List[str], HashSetTable]:
        """Tokenise the query and precompute per-term idf.

        Args:
            query: Raw query string.

        Returns:
            ``(unique_terms, idf_by_term)`` in first-occurrence order.

        Complexity:
            Time O(len(query) + sum |postings(t)|) -- the df computation over a
            candidate view walks postings; space O(number of query terms).
        """
        tokenizer = self.index.tokenizer  # type: ignore[attr-defined]
        tokens = tokenizer.tokenize(query)
        n_docs = self.index.N
        terms: List[str] = []
        idf_by_term: HashSetTable = HashSetTable()
        for term in tokens:
            if idf_by_term.contains(term):
                continue
            df = self.index.document_frequency(term)
            if df == 0:
                continue
            idf_by_term.put(term, self.idf(df, n_docs))
            terms.append(term)
        return terms, idf_by_term

    # -- search ------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[RankedItem]:
        """Rank documents for ``query`` and return the top ``k``.

        Args:
            query: Raw query string.
            k: Number of results.

        Returns:
            ``(score, doc_id)`` pairs, best first.

        Complexity:
            Time O(sum_t |postings(t)| + P log K) where P is the number of
            postings actually scored -- ``O(P log K)`` overall.  Space O(D + K)
            for the accumulation array (D = corpus size) and the heap.

        Note:
            The accumulation array is a plain ``list`` of floats indexed by
            ``doc_id``; that is a dense array, not a hash map, and it is what
            makes the scoring loop cache-friendly.
        """
        terms, idf_by_term = self._prepare_query(query)
        n_docs = self.index.N
        if not terms or n_docs == 0:
            self.scored_documents = 0
            self.last_stats = {"terms": 0.0, "scored": 0.0, "postings_visited": 0.0}
            return []

        scores = [0.0] * n_docs
        doc_lengths: List[int] = self.index.doc_lengths  # type: ignore[attr-defined]
        average_length: float = self.index.average_document_length  # type: ignore[attr-defined]
        candidates = getattr(self.index, "candidates", None)
        pool = getattr(self.index, "pool", None)
        is_candidate = candidates.contains if candidates is not None else None
        postings_visited = 0
        scored = 0

        for term in terms:
            idf = idf_by_term[term]
            for doc_id, tf in self.index.postings_for(term):
                postings_visited += 1
                if is_candidate is not None and not is_candidate(doc_id):
                    continue
                contribution = self.term_score(tf, doc_lengths[doc_id], average_length, idf)
                if contribution:
                    scores[doc_id] += contribution
                    scored += 1
                    if pool is not None:
                        pool.mark(doc_id)

        # When scoring against a pool bitmap, iteration is bounded by the
        # candidates actually touched rather than the whole corpus.
        heap = TopKHeap(k)
        if pool is not None:
            for doc_id in pool.ids():
                score = scores[doc_id]
                if score > 0.0:
                    heap.push(score, doc_id)
        else:
            for doc_id in range(n_docs):
                score = scores[doc_id]
                if score > 0.0:
                    heap.push(score, doc_id)

        self.scored_documents = scored
        self.last_stats = {
            "terms": float(len(terms)),
            "scored": float(scored),
            "postings_visited": float(postings_visited),
        }
        return heap.get_top_k()

    def score_document(self, query: str, doc_id: int) -> float:
        """Exact BM25 score of one document via document-at-a-time scoring.

        Args:
            query: Raw query string.
            doc_id: Document identifier.

        Returns:
            The score (0 when the document shares no term with the query).

        Complexity:
            Time O(number of query terms x cost of a posting lookup) -- for
            this index representation that is O(T_q * postings(t)) worst case;
            with a dense term-id map it would be O(T_q log P).  Space O(1).
        """
        terms, idf_by_term = self._prepare_query(query)
        if not terms:
            return 0.0
        total = 0.0
        frequencies: HashSetTable = HashSetTable()
        for term in terms:
            for posting_doc_id, tf in self.index.postings_for(term):
                if posting_doc_id == doc_id:
                    frequencies.put(term, tf)
                    break
        if len(frequencies) == 0:
            return 0.0
        doc_lengths: List[int] = self.index.doc_lengths  # type: ignore[attr-defined]
        average_length: float = self.index.average_document_length  # type: ignore[attr-defined]
        for term, tf in frequencies.items():
            total += self.term_score(tf, doc_lengths[doc_id], average_length, idf_by_term[term])
        return total

    # -- explanation -------------------------------------------------------

    def explain(self, query: str, doc_id: int) -> List[Tuple[str, int, float, float]]:
        """Per-term breakdown for one document, for hand-tracing in the report.

        Args:
            query: Raw query string.
            doc_id: Document identifier.

        Returns:
            ``(term, tf, idf, contribution)`` rows in query-term order, with
            zero-contribution rows included so the trace is complete.

        Complexity:
            Time O(sum_t |postings(t)|); space O(number of query terms).
        """
        terms, idf_by_term = self._prepare_query(query)
        doc_lengths: List[int] = self.index.doc_lengths  # type: ignore[attr-defined]
        average_length: float = self.index.average_document_length  # type: ignore[attr-defined]
        document_length = doc_lengths[doc_id] if 0 <= doc_id < len(doc_lengths) else 0
        rows: List[Tuple[str, int, float, float]] = []
        for term in terms:
            tf = 0
            for posting_doc_id, posting_tf in self.index.postings_for(term):
                if posting_doc_id == doc_id:
                    tf = posting_tf
                    break
            idf = idf_by_term[term]
            rows.append(
                (term, tf, idf, self.term_score(tf, document_length, average_length, idf))
            )
        return rows

    def explain_text(self, query: str, doc_id: int) -> str:
        """Human-readable version of :meth:`explain`.

        Args:
            query: Raw query string.
            doc_id: Document identifier.

        Returns:
            A multi-line report with the formula and each term's numbers.

        Complexity:
            Time O(sum_t |postings(t)|); space O(number of query terms).
        """
        rows = self.explain(query, doc_id)
        n_docs = self.index.N
        average_length = self.index.average_document_length  # type: ignore[attr-defined]
        doc_lengths: List[int] = self.index.doc_lengths  # type: ignore[attr-defined]
        dl = doc_lengths[doc_id] if 0 <= doc_id < len(doc_lengths) else 0
        lines = [
            f"BM25 explanation  query={query!r}  doc_id={doc_id}",
            f"  N={n_docs}  dl={dl}  avgdl={average_length:.4f}  k1={self.k1}  b={self.b}",
            f"  score = sum idf(t) * tf*(k1+1) / (tf + k1*(1-b + b*dl/avgdl))",
        ]
        total = 0.0
        for term, tf, idf, contribution in rows:
            total += contribution
            lines.append(
                f"  {term:<18s} tf={tf:<4d} idf={idf:.6f}  contribution={contribution:.6f}"
            )
        lines.append(f"  total = {total:.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return f"BM25(N={self.index.N}, k1={self.k1}, b={self.b})"


def build_scorer(index: InvertedIndex) -> BM25:
    """Convenience factory used by the agent and the CLI.

    Args:
        index: A built inverted index.

    Returns:
        A scorer bound to the full index.

    Complexity: O(1).
    """
    return BM25(index)


__all__ = ["BM25", "build_scorer"]
