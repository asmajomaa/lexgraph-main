"""TF-IDF vectorizer producing L2-normalised sparse vectors.

Two passes over the corpus:

1. **Count** -- document frequencies for every token (a builder-local table,
   not index state).
2. **Build** -- select the top-``M`` terms by document frequency subject to
   ``min_df``/``max_df``, assign each a column, then emit one
   :class:`~structures.sparse_vector.SparseVector` per document using
   ``tf * idf`` with the smoothed idf ``ln((1 + N) / (1 + df)) + 1``.

Because every vector is L2-normalised, Euclidean K-Means on these vectors
behaves like spherical K-Means on cosine similarity::

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b = 2 - 2 cos(a, b)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from config import VECTORIZER_MAX_DF_RATIO, VECTORIZER_MIN_DF, VECTORIZER_VOCAB_SIZE
from core.tokenizer import Tokenizer
from data.documents import Document
from structures.hash_table import HashSetTable
from structures.sparse_vector import SparseVector


def idf_smoothed(document_frequency: int, n_documents: int) -> float:
    """Smoothed inverse document frequency.

    Args:
        document_frequency: Number of documents containing the term.
        n_documents: Corpus size.

    Returns:
        ``ln((1 + N) / (1 + df)) + 1``; always >= 1 so every term keeps weight
        and an unseen term cannot produce an infinite weight.

    Complexity:
        Time O(1); space O(1).
    """
    return math.log((1.0 + n_documents) / (1.0 + document_frequency)) + 1.0


class TFIDFVectorizer:
    """Corpus-fitted TF-IDF vectorizer over the top-M terms by document frequency.

    Attributes:
        vocab_size: Maximum vocabulary size ``M``.
        min_df: Minimum document frequency.
        max_df_ratio: Maximum document frequency as a fraction of ``N``.
        vocabulary: ``term -> column``.
        idf: ``column -> idf weight``, parallel to :attr:`index_to_term`.
        index_to_term: ``column -> term``.
        n_documents: Corpus size seen at fit time.

    Space: O(V + M) for the vocabulary, plus the vectors it emits.
    """

    __slots__ = (
        "vocab_size",
        "min_df",
        "max_df_ratio",
        "vocabulary",
        "idf",
        "index_to_term",
        "n_documents",
        "tokenizer",
    )

    def __init__(
        self,
        vocab_size: int = VECTORIZER_VOCAB_SIZE,
        min_df: int = VECTORIZER_MIN_DF,
        max_df_ratio: float = VECTORIZER_MAX_DF_RATIO,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        """Create an unfitted vectorizer.

        Args:
            vocab_size: Maximum vocabulary size.
            min_df: Minimum document frequency.
            max_df_ratio: Maximum document frequency ratio.
            tokenizer: Tokenizer; defaults to the engine default.

        Complexity: O(1).
        """
        self.vocab_size = vocab_size
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocabulary: HashSetTable = HashSetTable()
        self.idf: List[float] = []
        self.index_to_term: List[str] = []
        self.n_documents = 0
        self.tokenizer = tokenizer or Tokenizer()

    # -- fitting -----------------------------------------------------------

    def fit(self, documents: Sequence[Document]) -> "TFIDFVectorizer":
        """Learn the vocabulary and idf weights.

        Args:
            documents: Corpus to fit on.

        Returns:
            ``self``, fitted.

        Complexity:
            Time O(total tokens + V log V) for vocabulary ranking;
            space O(V + M).
        """
        document_frequency: HashSetTable = HashSetTable()
        tokenize = self.tokenizer.tokenize
        for document in documents:
            seen: HashSetTable = HashSetTable()
            for term in tokenize(document.text):
                if seen.contains(term):
                    continue
                seen.put(term, 1)
                document_frequency.put(term, (document_frequency.get(term, 0) or 0) + 1)

        n_documents = len(documents)
        self.n_documents = n_documents
        max_df = max(1, int(self.max_df_ratio * n_documents)) if n_documents else 1

        eligible: List[Tuple[int, str]] = []
        for term, frequency in document_frequency.items():
            if frequency < self.min_df or frequency > max_df:
                continue
            eligible.append((-frequency, term))

        # Rank by descending document frequency, then alphabetically so the
        # vocabulary is deterministic regardless of hash iteration order.
        eligible.sort()
        selected = eligible[: self.vocab_size]

        self.vocabulary = HashSetTable()
        self.index_to_term = []
        self.idf = []
        for column, (negative_frequency, term) in enumerate(selected):
            self.vocabulary.put(term, column)
            self.index_to_term.append(term)
            self.idf.append(idf_smoothed(-negative_frequency, n_documents))
        return self

    @property
    def dimension(self) -> int:
        """Vocabulary size after fitting.

        Complexity: O(1).
        """
        return len(self.index_to_term)

    # -- transformation ----------------------------------------------------

    def transform_text(self, text: str) -> SparseVector:
        """Encode one raw text as a normalised TF-IDF vector.

        Args:
            text: Raw text.

        Returns:
            A unit-length sparse vector of length :attr:`dimension` (or the
            zero vector when no term is in the vocabulary).

        Complexity:
            Time O(len(text) + u log u) for u unique in-vocabulary terms
            (the log comes from sorting the pairs); space O(u).
        """
        counts: HashSetTable = HashSetTable()
        vocabulary = self.vocabulary
        for term in self.tokenizer.tokenize(text):
            column = vocabulary.get(term)
            if column is None:
                continue
            counts.put(column, (counts.get(column, 0.0) or 0.0) + 1.0)

        if len(counts) == 0:
            return SparseVector.zeros(self.dimension)

        # log(1 + tf) damps repetition exactly like the classical lnc.ltc
        # weighting; combined with the smoothed idf it keeps single-occurrence
        # terms comparable across document lengths.
        pairs: List[Tuple[int, float]] = []
        for column, raw_tf in counts.items():
            weight = (1.0 + math.log(raw_tf)) * self.idf[column]
            if weight:
                pairs.append((column, weight))
        pairs.sort()
        norm = math.sqrt(sum(weight * weight for _column, weight in pairs))
        if norm == 0.0:
            return SparseVector.zeros(self.dimension)
        return SparseVector(
            self.dimension,
            [column for column, _weight in pairs],
            [weight / norm for _column, weight in pairs],
        )

    def transform(self, documents: Sequence[Document]) -> List[SparseVector]:
        """Encode a document sequence.

        Args:
            documents: Documents already used at fit time (or new ones).

        Returns:
            One vector per document, in order.

        Complexity:
            Time O(sum of tokenisation costs); space O(total nnz).
        """
        return [self.transform_text(document.text) for document in documents]

    def to_state(self) -> Dict[str, object]:
        """Serialise the fitted state.

        Returns:
            A picklable mapping.  ``vocabulary`` is flattened to a list of
            ``(term, column)`` pairs so the cache payload carries no custom
            class besides the ones ``engine.py`` documents; ``from_state``
            rebuilds the :class:`HashSetTable`.

        Complexity:
            Time O(M); space O(M).
        """
        return {
            "vocab_size": self.vocab_size,
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "vocabulary": list(self.vocabulary.items()),
            "idf": self.idf,
            "index_to_term": self.index_to_term,
            "n_documents": self.n_documents,
            "tokenizer": {
                "min_length": self.tokenizer.min_length,
                "remove_stop_words": self.tokenizer.remove_stop_words,
                "use_stemmer": self.tokenizer.use_stemmer,
            },
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "TFIDFVectorizer":
        """Rebuild from :meth:`to_state` output.

        Args:
            state: Output of :meth:`to_state`.

        Returns:
            A fitted vectorizer.

        Complexity:
            Time O(M); space O(M).
        """
        tokenizer_state = state["tokenizer"]  # type: ignore[index]
        vectorizer = cls(
            vocab_size=int(state["vocab_size"]),  # type: ignore[arg-type]
            min_df=int(state["min_df"]),  # type: ignore[arg-type]
            max_df_ratio=float(state["max_df_ratio"]),  # type: ignore[arg-type]
            tokenizer=Tokenizer(
                min_length=int(tokenizer_state["min_length"]),  # type: ignore[index]
                remove_stop_words=bool(  # type: ignore[index]
                    tokenizer_state["remove_stop_words"]
                ),
                use_stemmer=bool(tokenizer_state["use_stemmer"]),  # type: ignore[index]
            ),
        )
        vocabulary = HashSetTable()
        vocabulary.bulk_put(state["vocabulary"])  # type: ignore[arg-type]
        vectorizer.vocabulary = vocabulary
        vectorizer.idf = list(state["idf"])  # type: ignore[arg-type]
        vectorizer.index_to_term = list(state["index_to_term"])  # type: ignore[arg-type]
        vectorizer.n_documents = int(state["n_documents"])  # type: ignore[arg-type]
        return vectorizer

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return (
            f"TFIDFVectorizer(dimension={self.dimension}, "
            f"n_documents={self.n_documents})"
        )


__all__ = ["TFIDFVectorizer", "idf_smoothed"]
