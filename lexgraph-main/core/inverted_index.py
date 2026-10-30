"""Inverted index built on the project's own hash table.

Layout
------
``term -> [(doc_id, tf), ...]`` in a :class:`~structures.hash_table.PostingsTable`,
plus three parallel arrays indexed by ``doc_id``:

* ``doc_lengths[i]`` -- number of indexed tokens in document *i*
* ``labels[i]``      -- ground-truth class, used only for clustering evaluation
* ``sources[i]``     -- dataset tag, used only for display

``avgdl``, ``N`` and ``total_length`` are derived once at build time.  Document
frequencies are implicit in the postings length, so a lookup is O(1) and there
is no second table to keep consistent.

Candidate restriction
---------------------
The ML-accelerated path needs BM25 restricted to a cluster.  Rather than
building a second index, :meth:`InvertedIndex.restrict` wraps a doc-id
membership table and skips postings that are not members.  The membership test
is O(1) and the scan is still linear in the postings length, so pruning saves
work on *scoring*, not on postings traversal -- an important honest caveat that
lands in the report.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from core.tokenizer import Tokenizer, term_frequencies
from data.documents import Document
from structures.hash_table import HashSetTable, PostingsTable


class CandidateSet:
    """O(1)-membership doc-id filter for the pruned search path.

    Backed by a **dense bitmap** (``bytearray``) of length ``N``.  A hash table
    would also be O(1) expected, but every membership test would pay a
    polynomial hash over a stringified integer -- and the pruned scorer tests
    membership once per posting visited.

    The bitmap can be populated two ways:

    * :meth:`add_range`-style bulk marking from a cluster membership list, in
      ``O(c)``; or
    * :meth:`mark` from the scorer itself, in ``O(P)``, which turns a pruned
      search into "visit each posting once, mark it, keep or drop it" and
      removes the separate pool pass entirely.

    The scorer uses the second form because cluster pools reach 30% of the
    corpus; marking only the postings a query touches is strictly cheaper than
    marking a whole cluster.

    Attributes:
        _bits: Membership flags indexed by ``doc_id``.
        _count: Number of documents flagged.

    Space: O(N) bytes (~19 kB at full corpus size).
    """

    __slots__ = ("_bits", "_count")

    def __init__(self, doc_ids: Iterable[int] = (), size: int = 0) -> None:
        """Build the set.

        Args:
            doc_ids: Candidate document identifiers.
            size: Explicit size; defaults to ``max(doc_ids) + 1``.

        Complexity:
            Time O(c + N); space O(N) bytes.
        """
        materialised = list(doc_ids)
        limit = size
        for doc_id in materialised:
            if doc_id + 1 > limit:
                limit = doc_id + 1
        self._bits = bytearray(limit)
        self._count = 0
        for doc_id in materialised:
            if 0 <= doc_id < limit and not self._bits[doc_id]:
                self._bits[doc_id] = 1
                self._count += 1

    @classmethod
    def empty(cls, size: int) -> "CandidateSet":
        """An empty candidate set sized for a corpus of ``size`` documents.

        Args:
            size: Corpus size.

        Returns:
            An empty :class:`CandidateSet`.

        Complexity:
            Time O(N); space O(N) bytes.
        """
        return cls((), size=size)

    def mark(self, doc_id: int) -> bool:
        """Flag ``doc_id`` as a candidate.

        Args:
            doc_id: Document identifier.

        Returns:
            ``True`` when it was newly flagged.

        Complexity:
            Time O(1); space O(1).
        """
        if 0 <= doc_id < len(self._bits) and not self._bits[doc_id]:
            self._bits[doc_id] = 1
            self._count += 1
            return True
        return False

    def contains(self, doc_id: int) -> bool:
        """Membership test.

        Args:
            doc_id: Document identifier.

        Returns:
            ``True`` when the id is a candidate.

        Complexity:
            Time O(1); space O(1).
        """
        return 0 <= doc_id < len(self._bits) and self._bits[doc_id] == 1

    def __len__(self) -> int:
        """Number of distinct candidates.

        Complexity: O(1).
        """
        return self._count

    def ids(self) -> List[int]:
        """Candidate ids as a sorted list.

        Returns:
            Ascending ids.

        Complexity:
            Time O(N); space O(c).
        """
        return [index for index, flag in enumerate(self._bits) if flag]


class InvertedIndex:
    """Inverted index, document statistics and corpus metadata.

    Attributes:
        postings: The term table.
        doc_lengths: Tokens per document, indexed by ``doc_id``.
        labels: Ground-truth label per document.
        sources: Dataset tag per document.
        external_ids: Presentation id per document.
        titles: Title per document.
        snippets: Snippet per document.
        tokenizer: The tokenizer used at build time.
        total_length: Sum of ``doc_lengths``.

    Space: O(T + postings) where T is the vocabulary size.
    """

    __slots__ = (
        "postings",
        "doc_lengths",
        "labels",
        "sources",
        "external_ids",
        "titles",
        "snippets",
        "tokenizer",
        "total_length",
        "name",
    )

    def __init__(self, tokenizer: Optional[Tokenizer] = None, name: str = "") -> None:
        """Create an empty index.

        Args:
            tokenizer: Tokenizer to use for queries; defaults to the build
                default so that query and document tokenisation always agree.
            name: Human-readable index name (``"newsgroups"``/``"legal"``).

        Complexity: O(1).
        """
        self.postings = PostingsTable()
        self.doc_lengths: List[int] = []
        self.labels: List[str] = []
        self.sources: List[str] = []
        self.external_ids: List[str] = []
        self.titles: List[str] = []
        self.snippets: List[str] = []
        self.tokenizer = tokenizer or Tokenizer()
        self.total_length = 0
        self.name = name

    # -- construction ------------------------------------------------------

    def add_document(self, document: Document) -> None:
        """Tokenise and index one document.

        Args:
            document: The document to add; its ``doc_id`` must equal
                ``len(self.doc_lengths)`` so that arrays stay dense.

        Raises:
            ValueError: If ``document.doc_id`` does not match the expected
                next id.

        Complexity:
            Time O(len(text)) plus O(u) postings appends for u unique terms;
            space O(u).
        """
        expected = len(self.doc_lengths)
        if document.doc_id != expected:
            raise ValueError(
                f"doc_id {document.doc_id} does not match index position {expected}"
            )
        tokens = self.tokenizer.tokenize(document.text)
        frequencies = term_frequencies(tokens)
        doc_id = document.doc_id
        for term, frequency in frequencies:
            self.postings.add_posting(term, doc_id, frequency)
        length = len(tokens)
        self.doc_lengths.append(length)
        self.total_length += length
        self.labels.append(document.label)
        self.sources.append(document.source)
        self.external_ids.append(document.external_id)
        self.titles.append(document.title)
        self.snippets.append(document.snippet)

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        tokenizer: Optional[Tokenizer] = None,
        name: str = "",
    ) -> "InvertedIndex":
        """Build an index from a document sequence.

        Args:
            documents: Documents with dense ids ``0..n-1``.
            tokenizer: Tokenizer to use.
            name: Index name.

        Returns:
            The populated index.

        Complexity:
            Time O(total tokens); space O(total unique postings).
        """
        index = cls(tokenizer, name)
        for document in documents:
            index.add_document(document)
        return index

    # -- statistics --------------------------------------------------------

    @property
    def N(self) -> int:
        """Number of documents.

        Complexity: O(1).
        """
        return len(self.doc_lengths)

    @property
    def average_document_length(self) -> float:
        """Mean document length in tokens (``avgdl``).

        Returns:
            ``total_length / N``, or 0 for an empty index.

        Complexity: O(1).
        """
        count = self.N
        return self.total_length / count if count else 0.0

    def document_frequency(self, term: str) -> int:
        """Number of documents containing ``term``.

        Args:
            term: Vocabulary term.

        Returns:
            Document frequency.

        Complexity:
            Time O(1) expected; space O(1).
        """
        return self.postings.document_frequency(term)

    def postings_for(self, term: str) -> list:
        """Postings list for ``term``.

        Args:
            term: Vocabulary term.

        Returns:
            ``[(doc_id, tf), ...]`` (the live list, not a copy).

        Complexity:
            Time O(1) expected; space O(1).
        """
        return self.postings.get_postings(term)

    def vocabulary_size(self) -> int:
        """Number of distinct indexed terms.

        Complexity: O(1).
        """
        return len(self.postings)

    def terms(self) -> Iterator[str]:
        """Iterate the vocabulary.

        Yields:
            Terms.

        Complexity:
            Time O(T + capacity); space O(1).
        """
        return self.postings.keys()

    def collection_statistics(self) -> Dict[str, float]:
        """Summary statistics for the report and the web UI.

        Returns:
            Mapping with ``N``, ``avgdl``, ``vocabulary``, ``tokens``,
            ``postings``, ``load_factor``, ``collisions`` (lifetime insert
            collisions, including rehashes) and ``chain_collisions`` (entries
            sharing a bucket in the final table).

        Complexity:
            Time O(T + capacity) to count postings and scan buckets;
            space O(1).
        """
        postings_total = 0
        for values in self.postings.values():
            postings_total += len(values)  # type: ignore[arg-type]
        return {
            "N": float(self.N),
            "avgdl": self.average_document_length,
            "vocabulary": float(self.vocabulary_size()),
            "tokens": float(self.total_length),
            "postings": float(postings_total),
            "load_factor": self.postings.load_factor,
            "collisions": float(self.postings.collisions),
            "chain_collisions": float(self.postings.chain_collisions),
        }

    # -- candidate restriction --------------------------------------------

    def restrict(
        self,
        candidates: object,
        pool: Optional[CandidateSet] = None,
    ) -> "RestrictedView":
        """Wrap this index with a candidate filter.

        Args:
            candidates: Anything exposing ``contains(doc_id) -> bool`` -- a
                :class:`CandidateSet` or a :class:`ClusterFilter`.
            pool: Optional empty bitmap the scorer fills with the ids it
                actually scored, so the final top-K pass touches only those
                instead of the whole corpus.  Defaults to an empty bitmap of
                corpus size.

        Returns:
            A read-only view that :class:`core.bm25.BM25` can score.

        Complexity:
            Time O(N) to allocate the pool bitmap; space O(N) bytes.
        """
        return RestrictedView(self, candidates, pool)

    # -- serialisation -----------------------------------------------------

    def to_state(self) -> dict:
        """Serialise to a plain structure suitable for ``pickle``.

        Returns:
            State mapping.  Title, snippet, external id, label and source are
            stored verbatim; postings are stored as a list of
            ``(term, postings)`` pairs because the custom table is not itself
            picklable without exposing internals.

        Complexity:
            Time O(T + postings); space O(T + postings).
        """
        return {
            "name": self.name,
            "doc_lengths": self.doc_lengths,
            "labels": self.labels,
            "sources": self.sources,
            "external_ids": self.external_ids,
            "titles": self.titles,
            "snippets": self.snippets,
            "tokenizer": {
                "min_length": self.tokenizer.min_length,
                "remove_stop_words": self.tokenizer.remove_stop_words,
                "use_stemmer": self.tokenizer.use_stemmer,
            },
            "postings": list(self.postings.items()),
        }

    @classmethod
    def from_state(cls, state: dict) -> "InvertedIndex":
        """Rebuild an index from :meth:`to_state` output.

        Args:
            state: Output of :meth:`to_state`.

        Returns:
            A populated index.

        Complexity:
            Time O(T + postings); space O(T + postings).
        """
        tokenizer_state = state["tokenizer"]
        index = cls(
            Tokenizer(
                min_length=int(tokenizer_state["min_length"]),
                remove_stop_words=bool(tokenizer_state["remove_stop_words"]),
                use_stemmer=bool(tokenizer_state["use_stemmer"]),
            ),
            str(state.get("name", "")),
        )
        index.doc_lengths = list(state["doc_lengths"])
        index.labels = list(state["labels"])
        index.sources = list(state["sources"])
        index.external_ids = list(state["external_ids"])
        index.titles = list(state["titles"])
        index.snippets = list(state["snippets"])
        index.total_length = sum(index.doc_lengths)
        index.postings.bulk_put(state["postings"])
        return index

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return (
            f"InvertedIndex(N={self.N}, vocab={self.vocabulary_size()}, "
            f"avgdl={self.average_document_length:.1f})"
        )


class ClusterFilter:
    """O(1) membership test of "is this document in one of the probed clusters?".

    This is the filter the ML path actually needs, and it is materially cheaper
    than materialising a candidate set per query.  Two precomputed arrays do
    the work:

    * ``cluster_of`` -- ``doc_id -> cluster``, built **once** when the agent is
      assembled (O(N) once, not per query);
    * ``mask`` -- one byte per cluster, set for the clusters being probed, built
      per query in O(k) where k = 20.

    A membership test is then two array indices, no hashing and no allocation
    proportional to the corpus.  At 18 719 documents and 5 000-term vectors,
    building a 5 597-entry candidate set per query cost ~0.87 ms, which is more
    than the entire baseline search; this filter costs ~0.01 ms.

    Attributes:
        cluster_of: ``doc_id -> cluster``.
        mask: ``cluster -> 0/1`` for the probed clusters.
        probed: The probed cluster ids, for reporting.

    Space: O(N) for ``cluster_of`` (shared across queries) plus O(k) per query.
    """

    __slots__ = ("cluster_of", "mask", "probed")

    def __init__(
        self,
        cluster_of: Sequence[int],
        probed: Sequence[int],
        n_clusters: int = 0,
    ) -> None:
        """Create a filter.

        Args:
            cluster_of: Per-document cluster assignment (shared, not copied).
            probed: Cluster ids to accept.
            n_clusters: Total number of clusters.  The mask must be sized to
                the *cluster count*, not to the largest probed id: a document
                in an unprobed cluster with a high id still has to be testable.

        Complexity:
            Time O(k + n_clusters); space O(n_clusters).
        """
        self.cluster_of = cluster_of
        self.probed = list(probed)
        limit = n_clusters
        for cluster in self.probed:
            if cluster + 1 > limit:
                limit = cluster + 1
        self.mask = bytearray(limit)
        for cluster in self.probed:
            if 0 <= cluster < limit:
                self.mask[cluster] = 1

    def contains(self, doc_id: int) -> bool:
        """Whether ``doc_id`` belongs to a probed cluster.

        Args:
            doc_id: Document identifier.

        Returns:
            ``True`` when the document is a candidate.

        Complexity:
            Time O(1); space O(1).
        """
        if not (0 <= doc_id < len(self.cluster_of)):
            return False
        cluster = self.cluster_of[doc_id]
        return cluster < len(self.mask) and self.mask[cluster] == 1

    def pool_size(self) -> int:
        """Number of documents in the probed clusters.

        Returns:
            Candidate pool size.

        Complexity:
            Time O(N); space O(1).
        """
        mask = self.mask
        cluster_of = self.cluster_of
        total = 0
        for cluster in cluster_of:
            if cluster < len(mask) and mask[cluster]:
                total += 1
        return total

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return f"ClusterFilter(probed={self.probed})"


class RestrictedView:
    """A candidate-filtered, read-only projection of an index.

    Attributes:
        index: The underlying index.
        candidates: The candidate set.
        pool: Bitmap the scorer fills with documents it actually scored.
        postings: Alias of ``index.postings`` for a uniform scorer interface.

    Space: O(1) beyond the referenced index and the two bitmaps.
    """

    __slots__ = ("index", "candidates", "pool", "postings")

    def __init__(
        self,
        index: InvertedIndex,
        candidates: object,
        pool: Optional[CandidateSet] = None,
    ) -> None:
        """Create the view.

        Args:
            index: Underlying index.
            candidates: Candidate filter exposing ``contains``.
            pool: Optional scoring bitmap; defaults to an empty one.

        Complexity:
            Time O(N) for the default bitmap; space O(N) bytes.
        """
        self.index = index
        self.candidates = candidates
        self.pool = pool if pool is not None else CandidateSet.empty(index.N)
        self.postings = index.postings

    @property
    def N(self) -> int:
        """Document count of the underlying corpus.

        Complexity: O(1).
        """
        return self.index.N

    @property
    def average_document_length(self) -> float:
        """``avgdl`` of the underlying corpus.

        Complexity: O(1).
        """
        return self.index.average_document_length

    @property
    def doc_lengths(self) -> List[int]:
        """Document lengths of the underlying corpus.

        Complexity: O(1).
        """
        return self.index.doc_lengths

    @property
    def tokenizer(self) -> Tokenizer:
        """Tokenizer shared with the underlying index.

        Query tokenisation must match document tokenisation, so the view never
        carries its own.

        Complexity: O(1).
        """
        return self.index.tokenizer

    def document_frequency(self, term: str) -> int:
        """Document frequency of ``term`` across the **whole** corpus.

        Args:
            term: Vocabulary term.

        Returns:
            Global document frequency.

        Complexity:
            Time O(1) expected; space O(1).

        Note:
            The pruned scorer deliberately uses the global df rather than a
            candidate-local one.  Two reasons: (1) recomputing df over a
            multi-thousand-document candidate set costs more than the scoring
            it is meant to accelerate, and (2) a global idf keeps the
            pruned ranking on the *same scale* as the baseline, so the
            recall comparison in the report is a pure pruning effect and not a
            side effect of re-weighting.  Both are measured in the benchmarks.
        """
        return self.index.postings.document_frequency(term)

    def postings_for(self, term: str) -> list:
        """Postings for ``term`` (unfiltered; the scorer filters).

        Args:
            term: Vocabulary term.

        Returns:
            ``[(doc_id, tf), ...]``.

        Complexity:
            Time O(1); space O(1).
        """
        return self.index.postings.get_postings(term)

    def collection_statistics(self) -> Dict[str, float]:
        """Same shape as :meth:`InvertedIndex.collection_statistics`.

        Returns:
            Statistics with ``N`` replaced by the candidate count.

        Complexity:
            Time O(T); space O(1).
        """
        stats = self.index.collection_statistics()
        stats["N"] = float(len(self.candidates))
        stats["corpus_N"] = float(self.index.N)
        return stats

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return f"RestrictedView(candidates={len(self.candidates)}, N={self.N})"


__all__ = ["CandidateSet", "InvertedIndex", "RestrictedView"]
