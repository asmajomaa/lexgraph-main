"""Engine assembly, caching and cache invalidation.

The CLI, the web API and the benchmarks all need the same wired-up object
graph: two inverted indexes, a citation graph, a fitted vectorizer and a
fitted K-Means with its document-to-cluster map.  This module builds it once
and persists it.

Caching format
--------------
A single ``pickle`` file (``cache/engine.pkl``) holding plain Python objects:
lists, tuples, strings and numbers.  Custom structures are serialised through
their ``to_state`` methods and rebuilt on load, so the cache carries no class
definitions and stays readable by ``pickletools``.  Pickle is used rather than
a hand-rolled text format because the payload is 1.4 M postings and 18 719
documents; a text format would triple the file size and add a parsing bug
surface for no benefit at this scale.  This is documented in the README.

A fingerprint of the build parameters (corpus size, vocabulary size, k, token
settings) is stored beside the payload, so changing ``config.py`` invalidates
the cache instead of silently serving a stale model.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Optional, Sequence, Tuple

from config import (
    CACHE_DIR,
    DEFAULT_BFS_DEPTH,
    KMEANS_K,
    KMEANS_MAX_ITER,
    KMEANS_SEED,
    VECTORIZER_MIN_DF,
    VECTORIZER_VOCAB_SIZE,
)
from core.agent import ReActAgent
from core.inverted_index import InvertedIndex
from core.tokenizer import Tokenizer
from data.load_legal_cases import LegalCorpus, load_legal_cases
from data.load_newsgroups import NewsgroupsCorpus, load_newsgroups
from data.documents import Document
from ml.kmeans import KMeans
from ml.vectorizer import TFIDFVectorizer
from structures.graph import CitationGraph

#: Cache file name inside :data:`config.CACHE_DIR`.
CACHE_FILE = "engine.pkl"

#: Bump when the on-disk schema changes.
CACHE_VERSION = 2


class Engine:
    """Everything the agent needs, plus the source corpora for display.

    Attributes:
        news: The newsgroups corpus (documents, categories, source tag).
        legal: The legal corpus (cases, edges, aliases).
        news_index: Inverted index over ``news``.
        legal_index: Inverted index over ``legal``.
        graph: Citation graph over ``legal``.
        vectorizer: Fitted TF-IDF vectorizer.
        kmeans: Fitted cluster model.
        clusters: ``news doc_id -> cluster``.
        agent: The wired :class:`ReActAgent`.
        built_at: Unix timestamp of the build.
        fingerprint: Build-parameter mapping used for cache validation.
    """

    __slots__ = (
        "news",
        "legal",
        "news_index",
        "legal_index",
        "graph",
        "vectorizer",
        "kmeans",
        "clusters",
        "agent",
        "built_at",
        "fingerprint",
    )

    def __init__(
        self,
        news: NewsgroupsCorpus,
        legal: LegalCorpus,
        news_index: InvertedIndex,
        legal_index: InvertedIndex,
        graph: CitationGraph,
        vectorizer: TFIDFVectorizer,
        kmeans: KMeans,
        clusters: List[int],
        fingerprint: Dict[str, object],
        verbose: bool = True,
    ) -> None:
        """Create the engine and wire the agent.

        Args:
            news: Newsgroups corpus.
            legal: Legal corpus.
            news_index: News index.
            legal_index: Legal index.
            graph: Citation graph.
            vectorizer: Fitted vectorizer.
            kmeans: Fitted clusterer.
            clusters: Per-document cluster assignment.
            fingerprint: Build parameters for cache validation.
            verbose: Whether the agent prints logs.

        Complexity: O(n) to build the agent's inverse cluster map.
        """
        self.news = news
        self.legal = legal
        self.news_index = news_index
        self.legal_index = legal_index
        self.graph = graph
        self.vectorizer = vectorizer
        self.kmeans = kmeans
        self.clusters = clusters
        self.fingerprint = fingerprint
        self.built_at = time.time()
        self.agent = ReActAgent(
            news_index=news_index,
            legal_index=legal_index,
            graph=graph,
            legal_corpus=legal,
            kmeans=kmeans,
            vectorizer=vectorizer,
            clusters=clusters,
            verbose=verbose,
        )

    def query(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        """Delegate to the agent.

        Args:
            *args: Positional arguments for :meth:`ReActAgent.run`.
            **kwargs: Keyword arguments for :meth:`ReActAgent.run`.

        Returns:
            The :class:`core.agent.AgentResult`.

        Complexity: see :meth:`ReActAgent.run`.
        """
        return self.agent.run(*args, **kwargs)  # type: ignore[arg-type]


def build_citation_graph(legal: LegalCorpus) -> Tuple[CitationGraph, int]:
    """Build the typed citation graph from a legal corpus.

    Args:
        legal: The legal corpus with resolved edges.

    Returns:
        ``(graph, skipped)`` where ``skipped`` counts edges whose endpoint is
        not a known case external id.

    Complexity:
        Time O(V + E); space O(V + E).
    """
    graph = CitationGraph()
    by_id = {document.doc_id: document.external_id for document in legal.documents}
    for doc_id, external_id in by_id.items():
        del doc_id
        graph.add_node(external_id)
    skipped = 0
    for source_id, target_id, edge_class in legal.edges:
        source = by_id.get(source_id)
        target = by_id.get(target_id)
        if source is None or target is None:
            skipped += 1
            continue
        graph.add_edge(source, target, edge_class)
    return graph, skipped


def _fingerprint(news_count: int, legal_count: int) -> Dict[str, object]:
    """Build the cache-validation fingerprint.

    Args:
        news_count: Number of newsgroups documents.
        legal_count: Number of legal cases.

    Returns:
        Parameter mapping.

    Complexity: O(1).
    """
    return {
        "version": CACHE_VERSION,
        "news": news_count,
        "legal": legal_count,
        "vocab": VECTORIZER_VOCAB_SIZE,
        "min_df": VECTORIZER_MIN_DF,
        "k": KMEANS_K,
        "max_iter": KMEANS_MAX_ITER,
        "seed": KMEANS_SEED,
        "tokenizer": "default-v1",
    }


def build_engine(
    verbose: bool = True,
    news_limit: Optional[int] = None,
    legal_limit: Optional[int] = None,
    progress: bool = True,
) -> Engine:
    """Load the corpora and build every structure from scratch.

    Args:
        verbose: Whether the agent prints logs.
        news_limit: Cap on newsgroups documents (for fast test builds).
        legal_limit: Cap on legal cases.
        progress: Whether to print phase timings to stdout.

    Returns:
        A ready :class:`Engine`.

    Complexity:
        Time O(total tokens + n log V + n * k * nnz * iters) -- the K-Means fit
        dominates at full scale; space O(index + vectors + centroids).
    """
    started = time.perf_counter()

    def report(label: str, mark: float) -> float:
        """Print a phase split when ``progress`` is set."""
        now = time.perf_counter()
        if progress:
            print(f"[BUILD] {label:<28s} {now - mark:7.2f}s")
        return now

    mark = started
    news = load_newsgroups(limit=news_limit)
    legal = load_legal_cases(limit=legal_limit)
    mark = report(
        f"load corpora ({len(news)} news / {len(legal)} cases)", mark
    )

    news_index = InvertedIndex.build(news.documents, Tokenizer(), name="newsgroups")
    mark = report(f"news index ({news_index.vocabulary_size()} terms)", mark)

    legal_index = InvertedIndex.build(legal.documents, Tokenizer(), name="legal")
    mark = report(f"legal index ({legal_index.vocabulary_size()} terms)", mark)

    graph, skipped = build_citation_graph(legal)
    mark = report(f"citation graph ({graph.node_count()}/{graph.edge_count()} V/E)", mark)

    vectorizer = TFIDFVectorizer(vocab_size=VECTORIZER_VOCAB_SIZE, tokenizer=Tokenizer())
    vectorizer.fit(news.documents)
    mark = report(f"tf-idf vocabulary ({vectorizer.dimension})", mark)

    vectors = vectorizer.transform(news.documents)
    mark = report("vectorise corpus", mark)

    kmeans = KMeans(k=KMEANS_K, max_iter=KMEANS_MAX_ITER, seed=KMEANS_SEED)
    kmeans.fit(vectors)
    mark = report(
        f"k-means (k={kmeans.k}, iters={kmeans.n_iter}, {kmeans.converged})", mark
    )
    del vectors

    fingerprint = _fingerprint(len(news), len(legal))
    engine = Engine(
        news=news,
        legal=legal,
        news_index=news_index,
        legal_index=legal_index,
        graph=graph,
        vectorizer=vectorizer,
        kmeans=kmeans,
        clusters=list(kmeans.labels),
        fingerprint=fingerprint,
        verbose=verbose,
    )
    if progress:
        print(f"[BUILD] {'total':<28s} {time.perf_counter() - started:7.2f}s  ({skipped} edges with unknown endpoints skipped)")
    return engine


def cache_path() -> str:
    """Absolute path of the cache file.

    Returns:
        Path string.

    Complexity: O(1).
    """
    return os.path.join(str(CACHE_DIR), CACHE_FILE)


def save_engine(engine: Engine, path: Optional[str] = None) -> str:
    """Persist an engine to disk.

    Args:
        engine: The engine to save.
        path: Target file; defaults to :func:`cache_path`.

    Returns:
        The path written.

    Complexity:
        Time O(index + vectors); space O(file size) transiently.
    """
    target = path or cache_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    payload = {
        "fingerprint": engine.fingerprint,
        "built_at": engine.built_at,
        "news_documents": [document.to_dict() for document in engine.news.documents],
        "news_categories": engine.news.categories,
        "news_source": engine.news.source,
        "news_root": engine.news.root,
        "legal_documents": [document.to_dict() for document in engine.legal.documents],
        "legal_edges": engine.legal.edges,
        "legal_names": engine.legal.name_by_id,
        "legal_keys": engine.legal.key_by_id,
        "legal_source": engine.legal.source,
        "legal_class_counts": engine.legal.class_counts,
        "legal_records": engine.legal.citation_records,
        "legal_unresolved": engine.legal.unresolved,
        "news_index": engine.news_index.to_state(),
        "legal_index": engine.legal_index.to_state(),
        "vectorizer": engine.vectorizer.to_state(),
        "kmeans": engine.kmeans.to_state(),
        "clusters": engine.clusters,
    }
    with open(target, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_engine(path: Optional[str] = None, verbose: bool = True) -> Engine:
    """Load a persisted engine.

    Args:
        path: Source file; defaults to :func:`cache_path`.
        verbose: Whether the agent prints logs.

    Returns:
        The rebuilt :class:`Engine`.

    Raises:
        FileNotFoundError: When the cache does not exist.

    Complexity:
        Time O(index + vectors); space O(index).
    """
    source = path or cache_path()
    with open(source, "rb") as handle:
        payload = pickle.load(handle)

    news_documents = [Document.from_dict(item) for item in payload["news_documents"]]
    legal_documents = [Document.from_dict(item) for item in payload["legal_documents"]]
    legal = LegalCorpus(
        legal_documents,
        [tuple(edge) for edge in payload["legal_edges"]],
        {int(key): value for key, value in payload["legal_names"].items()},
        {int(key): value for key, value in payload["legal_keys"].items()},
        payload["legal_source"],
    )
    legal.build_aliases()
    legal.class_counts = dict(payload["legal_class_counts"])
    legal.citation_records = int(payload["legal_records"])
    legal.unresolved = int(payload["legal_unresolved"])

    news = NewsgroupsCorpus(
        news_documents,
        list(payload["news_categories"]),
        payload["news_source"],
        payload["news_root"],
    )
    news_index = InvertedIndex.from_state(payload["news_index"])
    legal_index = InvertedIndex.from_state(payload["legal_index"])
    graph, _skipped = build_citation_graph(legal)
    vectorizer = TFIDFVectorizer.from_state(payload["vectorizer"])
    kmeans = KMeans.from_state(payload["kmeans"])

    return Engine(
        news=news,
        legal=legal,
        news_index=news_index,
        legal_index=legal_index,
        graph=graph,
        vectorizer=vectorizer,
        kmeans=kmeans,
        clusters=list(payload["clusters"]),
        fingerprint=dict(payload["fingerprint"]),
        verbose=verbose,
    )


def engine_is_stale(path: Optional[str] = None) -> bool:
    """Check whether the cached engine matches the current parameters.

    Args:
        path: Cache file; defaults to :func:`cache_path`.

    Returns:
        ``True`` when the cache is missing, unreadable or built with different
        parameters.

    Complexity:
        Time O(1) (only the fingerprint is unpickled); space O(1).
    """
    target = path or cache_path()
    if not os.path.exists(target):
        return True
    try:
        with open(target, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return True
    stored = payload.get("fingerprint", {})
    for key in ("version", "vocab", "min_df", "k", "max_iter", "seed", "tokenizer"):
        if stored.get(key) != _fingerprint(0, 0).get(key):
            return True
    return False


def get_engine(
    rebuild: bool = False,
    verbose: bool = True,
    news_limit: Optional[int] = None,
    legal_limit: Optional[int] = None,
) -> Engine:
    """Load the cached engine or build a fresh one.

    Args:
        rebuild: Force a rebuild even when a valid cache exists.
        verbose: Whether the agent prints logs.
        news_limit: Cap on newsgroups documents when building.
        legal_limit: Cap on legal cases when building.

    Returns:
        A ready :class:`Engine`.

    Complexity:
        Time O(1) when a valid cache is loaded (plus unpickling); otherwise
        the full :func:`build_engine` cost.
    """
    if not rebuild and not news_limit and not legal_limit and not engine_is_stale():
        return load_engine(verbose=verbose)
    engine = build_engine(
        verbose=verbose, news_limit=news_limit, legal_limit=legal_limit, progress=verbose
    )
    if not news_limit and not legal_limit:
        path = save_engine(engine)
        if verbose:
            print(f"[BUILD] cached engine -> {path}")
    return engine


__all__ = [
    "Engine",
    "build_engine",
    "build_citation_graph",
    "load_engine",
    "save_engine",
    "get_engine",
    "engine_is_stale",
    "cache_path",
]
