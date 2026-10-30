"""Benchmark 5: this implementation vs established Python implementations.

Every comparison uses the **same inputs and the same tokenisation**, so the
differences are the implementation and nothing else:

=================================  ==========================================
mine                               reference
=================================  ==========================================
custom hash table                  built-in ``dict``
custom heap                        ``heapq``
custom BM25                        ``rank_bm25.BM25Okapi`` and
                                   sklearn ``TfidfVectorizer`` + cosine
custom K-Means                     ``sklearn.cluster.KMeans``
custom BFS                         ``networkx.bfs_edges``
=================================  ==========================================

Memory is measured with ``tracemalloc`` around construction only.  The
conclusion is expected to be "mine is slower, but transparent and
customisable"; the tables report whatever the measurements show.
"""

from __future__ import annotations

import heapq
import random
import time
import tracemalloc
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.common import (
    PALETTE,
    figure_path,
    matplotlib_backend,
    style_axes,
    time_call,
    write_csv,
    write_markdown_table,
)
from config import BENCH_REPEATS, KMEANS_SEED
from core.bm25 import BM25
from core.inverted_index import InvertedIndex
from core.tokenizer import Tokenizer
from ml.kmeans import KMeans
from structures.graph import CitationGraph
from structures.hash_table import HashSetTable
from structures.min_heap import TopKHeap


def _peak_kib(builder) -> Tuple[float, object]:  # type: ignore[no-untyped-def]
    """Measure peak allocation of ``builder()`` in KiB.

    Args:
        builder: Zero-argument callable that returns a structure.

    Returns:
        ``(peak_kib, built_object)``.

    Complexity:
        Time and space of the builder plus ``tracemalloc`` overhead.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    built = builder()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0, built


def compare_hash_table(sample_size: int = 150_000, repeats: int = 3) -> Dict[str, object]:
    """Hash table vs built-in dict: insert, lookup, memory.

    Args:
        sample_size: Number of keys.
        repeats: Measurement repetitions.

    Returns:
        Result mapping.

    Complexity:
        Time O(sample_size) per measurement; space O(sample_size).
    """
    rng = random.Random(17)
    keys = [f"term{rng.randrange(10**9)}x{index}" for index in range(sample_size)]
    values = list(range(sample_size))
    probes = [keys[rng.randrange(sample_size)] for _ in range(sample_size)]

    def custom_insert() -> object:
        """Build the custom table."""
        table = HashSetTable()
        for key, value in zip(keys, values):
            table.put(key, value)
        return table

    def custom_lookup(table: object) -> None:
        """Look up every probe key in the custom table."""
        get = table.get  # type: ignore[attr-defined]
        for key in probes:
            get(key)

    def dict_lookup(reference: dict) -> None:
        """Look up every probe key in a dict."""
        get = reference.get
        for key in probes:
            get(key)

    insert_mean, _std = time_call(custom_insert, repeats)
    reference = dict(zip(keys, values))
    dict_build_mean, _ = time_call(lambda: dict(zip(keys, values)), repeats)
    dict_lookup_mean, _ = time_call(lambda: dict_lookup(reference), repeats)

    custom_peak, table = _peak_kib(custom_insert)
    lookup_mean, _ = time_call(lambda: custom_lookup(table), repeats)
    dict_peak, _ = _peak_kib(lambda: dict(zip(keys, values)))
    dict_peak = max(dict_peak, 1.0)

    print(
        f"[compare] hash: custom insert {insert_mean * 1000:.1f} ms vs dict build "
        f"{dict_build_mean * 1000:.1f} ms; lookup {lookup_mean * 1000:.1f} ms vs "
        f"{dict_lookup_mean * 1000:.1f} ms; peak {custom_peak:.0f} vs {dict_peak:.0f} KiB"
    )
    return {
        "n": sample_size,
        "insert_ms": insert_mean * 1000.0,
        "dict_build_ms": dict_build_mean * 1000.0,
        "lookup_ms": lookup_mean * 1000.0,
        "dict_lookup_ms": dict_lookup_mean * 1000.0,
        "peak_kib": custom_peak,
        "dict_peak_kib": dict_peak,
        # Lifetime insert collisions (each resize re-inserts every entry, so
        # this counts rehash work too); ``chain_collisions`` is the shape of
        # the final table and is what the uniform-hashing prediction compares
        # against.  Both are reported so neither can be misread.
        "collisions": table.collisions,  # type: ignore[attr-defined]
        "chain_collisions": table.chain_collisions,  # type: ignore[attr-defined]
        "load_factor": table.load_factor,  # type: ignore[attr-defined]
    }


def compare_heap(sample_size: int = 200_000, k: int = 10, repeats: int = 3) -> Dict[str, object]:
    """Bounded top-K heap vs ``heapq.nlargest``.

    Args:
        sample_size: Number of scored candidates.
        k: Top-K size.
        repeats: Measurement repetitions.

    Returns:
        Result mapping.

    Complexity:
        Time O(sample_size log k) per measurement; space O(k).
    """

    def scores() -> List[float]:
        """Deterministic candidate scores."""
        rng = random.Random(23)
        return [rng.random() for _ in range(sample_size)]

    data = scores()

    def custom() -> List[Tuple[float, int]]:
        """Drain a :class:`TopKHeap` over the candidates."""
        heap = TopKHeap(k)
        for doc_id, score in enumerate(data):
            heap.push(score, doc_id)
        return heap.get_top_k()

    def reference() -> List[Tuple[float, int]]:
        """``heapq.nlargest`` over the same candidates."""
        best = heapq.nlargest(k, ((score, doc_id) for doc_id, score in enumerate(data)))
        return best

    custom_mean, _ = time_call(custom, repeats)
    reference_mean, _ = time_call(reference, repeats)
    print(
        f"[compare] heap: custom {custom_mean * 1000:.2f} ms vs heapq "
        f"{reference_mean * 1000:.2f} ms"
    )
    return {
        "n": sample_size,
        "k": k,
        "custom_ms": custom_mean * 1000.0,
        "heapq_ms": reference_mean * 1000.0,
        "same_result": [score for score, _ in custom()] == [score for score, _ in reference()],
    }


def compare_bm25(engine, query_count: int = 25, k: int = 10) -> Dict[str, object]:
    """Custom BM25 vs ``rank_bm25`` and vs sklearn TF-IDF + cosine.

    Args:
        engine: A built engine.
        query_count: Number of queries sampled from titles.
        k: Top-K size.

    Returns:
        Result mapping.

    Complexity:
        Time O(queries x postings) for the custom scorer and
        O(queries x N x df) for rank_bm25 (it rescans); space O(corpus).
    """
    tokenizer: Tokenizer = engine.news_index.tokenizer
    documents = engine.news.documents
    tokenised = [tokenizer.tokenize(document.text) for document in documents]

    scorer = BM25(engine.news_index)
    queries = []
    rng = random.Random(29)
    while len(queries) < query_count:
        title = rng.choice(engine.news_index.titles)
        terms = tokenizer.tokenize(title)
        if len(terms) >= 2:
            queries.append(" ".join(terms[:3]))

    custom_ms_per_query, _ = time_call(
        lambda: [scorer.search(query, k) for query in queries], 2
    )
    custom_ms_per_query = custom_ms_per_query / len(queries) * 1000.0

    result: Dict[str, object] = {"custom_ms_per_query": custom_ms_per_query}

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        result["rank_bm25"] = "not installed"
    else:
        dense_ms, _ = time_call(lambda: BM25Okapi(tokenised), 1)
        reference = BM25Okapi(tokenised)
        reference_ms, _ = time_call(
            lambda: [reference.get_scores(query.split()) for query in queries], 2
        )
        reference_ms = reference_ms / len(queries) * 1000.0
        overlaps = []
        for query in queries:
            reference_scores = reference.get_scores(tokenizer.tokenize(query))
            order = sorted(range(len(reference_scores)), key=lambda i: (-reference_scores[i], i))
            reference_top = set(order[:k])
            custom_top = {doc_id for _score, doc_id in scorer.search(query, k)}
            overlaps.append(len(custom_top & reference_top) / k)
        result["rank_bm25"] = {
            "index_build_ms": dense_ms * 1000.0,
            "query_ms": reference_ms,
            "overlap_at_k": sum(overlaps) / len(overlaps),
        }
        print(
            f"[compare] bm25 vs rank_bm25: custom {custom_ms_per_query:.3f} ms/query vs "
            f"{reference_ms:.3f} ms/query; top-{k} overlap "
            f"{sum(overlaps) / len(overlaps) * 100:.1f}%"
        )

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        result["sklearn_tfidf"] = "not installed"
    else:
        texts = [document.text for document in documents]
        vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize, lowercase=False)
        fit_ms, _ = time_call(lambda: vectorizer.fit_transform(texts), 1)
        matrix = vectorizer.fit_transform(texts)
        from sklearn.metrics.pairwise import cosine_similarity

        def sklearn_queries() -> None:
            """Score every query against the matrix with cosine similarity."""
            for query in queries:
                cosine_similarity(vectorizer.transform([query]), matrix)

        query_ms, _ = time_call(sklearn_queries, 2)
        query_ms = query_ms / len(queries) * 1000.0
        overlaps = []
        for query in queries:
            scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            reference_top = set(order[:k])
            custom_top = {doc_id for _score, doc_id in scorer.search(query, k)}
            overlaps.append(len(custom_top & reference_top) / k)
        result["sklearn_tfidf"] = {
            "fit_ms": fit_ms * 1000.0,
            "query_ms": query_ms,
            "overlap_at_k": sum(overlaps) / len(overlaps),
        }
        print(
            f"[compare] bm25 vs sklearn tf-idf+cosine: custom {custom_ms_per_query:.3f} ms/query "
            f"vs {query_ms:.3f} ms/query; top-{k} overlap "
            f"{sum(overlaps) / len(overlaps) * 100:.1f}%"
        )
    return result


def compare_kmeans(engine, sample: int = 4000) -> Dict[str, object]:
    """Custom K-Means vs ``sklearn.cluster.KMeans`` on the same vectors.

    Args:
        engine: A build engine.
        sample: Number of documents to cluster (sklearn on the full corpus is
            far slower than the pure-Python run and adds no information).

    Returns:
        Result mapping.

    Complexity:
        Time O(sample x k x nnz x iters); space O(sample x M) dense.
    """
    vectors = engine.vectorizer.transform(engine.news.documents[:sample])
    truth = engine.news_index.labels[:sample]

    def custom_fit() -> KMeans:
        """Fit the custom clusterer."""
        return KMeans(k=20, max_iter=60, seed=KMEANS_SEED).fit(vectors)

    custom_mean, _ = time_call(custom_fit, 1)
    custom_model = custom_fit()

    from ml.evaluation import nmi

    result: Dict[str, object] = {
        "sample": sample,
        "custom_ms": custom_mean * 1000.0,
        "custom_inertia": custom_model.inertia,
        "custom_nmi": nmi(truth, custom_model.labels),
    }

    try:
        import numpy as np
        from sklearn.cluster import KMeans as SklearnKMeans
    except ImportError:
        result["sklearn"] = "not installed"
        return result

    dense = np.zeros((len(vectors), vectors[0].dimension))
    for row, vector in enumerate(vectors):
        for coordinate, value in zip(vector.indices, vector.values):
            dense[row, coordinate] = value

    def sklearn_fit() -> object:
        """Fit sklearn's K-Means with the same k, seed and iteration budget.

        ``n_init=10`` is the library's own default and the honest setting: with
        ``n_init=1`` sklearn stopped after 3 iterations at inertia 3889, a
        noticeably worse local optimum than k-means++ alone reaches, which
        would have inflated this project's apparent quality.  Comparing our
        single k-means++ run against sklearn's ten restarts is the *stricter*
        test, and the report says so.
        """
        model = SklearnKMeans(n_clusters=20, n_init=10, random_state=KMEANS_SEED, max_iter=60)
        return model.fit(dense)

    sklearn_mean, _ = time_call(sklearn_fit, 1)
    sklearn_model = sklearn_fit()
    result["sklearn"] = {
        "ms": sklearn_mean * 1000.0,
        "inertia": float(sklearn_model.inertia_),
        "nmi": nmi(truth, sklearn_model.labels_.tolist()),
    }
    print(
        f"[compare] kmeans: custom {custom_mean * 1000:.0f} ms "
        f"(inertia {custom_model.inertia:.1f}, "
        f"NMI {result['custom_nmi']:.4f}) vs sklearn {sklearn_mean * 1000:.0f} ms "
        f"(inertia {sklearn_model.inertia_:.1f}, NMI {result['sklearn']['nmi']:.4f})"
    )
    return result


def compare_bfs(engine) -> Dict[str, object]:
    """Custom BFS vs ``networkx.bfs_edges`` on the real citation graph.

    Args:
        engine: A built engine.

    Returns:
        Result mapping.

    Complexity:
        Time O(V + E) per traversal; space O(V).
    """
    graph = engine.graph
    seed = max(graph.nodes(), key=lambda node: graph.degree(node, "both"))

    def custom() -> int:
        """Full custom BFS from the seed."""
        return len(graph.bfs_traverse(seed, max_depth=10**9, direction="both"))

    # The very first traversal also assigns the graph's node index (one hash
    # per node, then a list index per visited test).  Timing it inside the
    # repeat loop mixes a one-off O(V) setup into a per-traversal average, and
    # that is what an earlier version reported as a 15-76 ms "cold" BFS with a
    # spread far wider than the machine's noise.  Measure the two separately:
    # the setup once, then the traversal on a warm graph.
    setup_started = time.perf_counter()
    custom()
    setup_ms = (time.perf_counter() - setup_started) * 1000.0
    custom_mean, custom_std = time_call(custom, BENCH_REPEATS)
    result: Dict[str, object] = {
        "seed": seed,
        "custom_ms": custom_mean * 1000.0,
        "custom_std_ms": custom_std * 1000.0,
        "first_call_ms": setup_ms,
    }

    try:
        import networkx as nx
    except ImportError:
        result["networkx"] = "not installed"
        return result

    directed = nx.DiGraph()
    for node in graph.nodes():
        directed.add_node(node)
    for node in graph.nodes():
        for neighbour, _etype in graph.neighbors(node, "out"):
            directed.add_edge(node, neighbour)
    undirected = directed.to_undirected()

    def reference() -> int:
        """networkx BFS over the undirected view."""
        return len(list(nx.bfs_edges(undirected, seed, depth_limit=None))) + 1

    reference_mean, reference_std = time_call(reference, BENCH_REPEATS)
    result["networkx"] = {
        "ms": reference_mean * 1000.0,
        "std_ms": reference_std * 1000.0,
        "reached": reference(),
    }
    result["custom_reached"] = custom()
    print(
        f"[compare] bfs: custom {custom_mean * 1000:.3f} ms/traversal "
        f"(first call {setup_ms:.3f} ms, includes the node index) "
        f"vs networkx {reference_mean * 1000:.3f} ms "
        f"(both reach {result['custom_reached']} nodes)"
    )
    return result


def run() -> Dict[str, object]:
    """Execute every comparison and write the tables.

    Returns:
        Aggregate mapping.

    Complexity:
        Time dominated by ``compare_bm25``'s rank_bm25 pass; space O(corpus).
    """
    from engine import get_engine

    engine = get_engine(verbose=False)
    results: Dict[str, object] = {}
    results["hash"] = compare_hash_table()
    results["heap"] = compare_heap()
    results["bm25"] = compare_bm25(engine)
    results["kmeans"] = compare_kmeans(engine)
    results["bfs"] = compare_bfs(engine)

    hash_result = results["hash"]
    heap_result = results["heap"]
    bm25_result = results["bm25"]
    kmeans_result = results["kmeans"]
    bfs_result = results["bfs"]

    write_csv(
        "table_5_comparison.csv",
        ["component", "mine", "reference", "ratio", "note"],
        [
            (
                "hash table insert",
                f"{hash_result['insert_ms']:.2f} ms",
                f"{hash_result['dict_build_ms']:.2f} ms",
                f"{hash_result['insert_ms'] / max(1e-9, hash_result['dict_build_ms']):.1f}x",
                f"{hash_result['n']} keys, load {hash_result['load_factor']:.2f}, "
                f"{hash_result['chain_collisions']} final-table collisions "
                f"({hash_result['collisions']} over the whole build)",
            ),
            (
                "hash table lookup",
                f"{hash_result['lookup_ms']:.2f} ms",
                f"{hash_result['dict_lookup_ms']:.2f} ms",
                f"{hash_result['lookup_ms'] / max(1e-9, hash_result['dict_lookup_ms']):.1f}x",
                f"{hash_result['n']} lookups",
            ),
            (
                "hash memory",
                f"{hash_result['peak_kib']:.0f} KiB",
                f"{hash_result['dict_peak_kib']:.0f} KiB",
                f"{hash_result['peak_kib'] / max(1e-9, hash_result['dict_peak_kib']):.2f}x",
                "peak via tracemalloc",
            ),
            (
                "top-K heap",
                f"{heap_result['custom_ms']:.2f} ms",
                f"{heap_result['heapq_ms']:.2f} ms",
                f"{heap_result['custom_ms'] / max(1e-9, heap_result['heapq_ms']):.1f}x",
                f"k={heap_result['k']}, identical scores: {heap_result['same_result']}",
            ),
        ],
    )

    bm25_rows: List[Tuple[object, ...]] = [
        ("custom BM25", f"{bm25_result['custom_ms_per_query']:.3f} ms/query", "1.000", "reference")
    ]
    rank = bm25_result.get("rank_bm25")
    if isinstance(rank, dict):
        bm25_rows.append(
            (
                "rank_bm25.BM25Okapi",
                f"{rank['query_ms']:.3f} ms/query",
                f"{bm25_result['custom_ms_per_query'] / max(1e-9, rank['query_ms']):.2f}x",
                f"top-10 overlap {rank['overlap_at_k'] * 100:.1f}%",
            )
        )
    sklearn_tfidf = bm25_result.get("sklearn_tfidf")
    if isinstance(sklearn_tfidf, dict):
        tfidf_ratio = bm25_result["custom_ms_per_query"] / max(
            1e-9, sklearn_tfidf["query_ms"]
        )
        bm25_rows.append(
            (
                "sklearn TF-IDF + cosine",
                f"{sklearn_tfidf['query_ms']:.3f} ms/query",
                f"{tfidf_ratio:.2f}x",
                f"top-10 overlap {sklearn_tfidf['overlap_at_k'] * 100:.1f}%",
            )
        )
    write_markdown_table(
        "table_5_bm25_comparison.md",
        "BM25 / TF-IDF ranking comparison (identical tokenisation)",
        ["implementation", "query time", "ratio vs mine", "agreement"],
        bm25_rows,
    )

    kmeans_rows: List[Tuple[object, ...]] = [
        (
            "custom K-Means",
            f"{kmeans_result['custom_ms']:.0f} ms",
            f"{kmeans_result['custom_inertia']:.1f}",
            f"{kmeans_result['custom_nmi']:.4f}",
        )
    ]
    sklearn_kmeans = kmeans_result.get("sklearn")
    if isinstance(sklearn_kmeans, dict):
        kmeans_rows.append(
            (
                "sklearn.cluster.KMeans",
                f"{sklearn_kmeans['ms']:.0f} ms",
                f"{sklearn_kmeans['inertia']:.1f}",
                f"{sklearn_kmeans['nmi']:.4f}",
            )
        )
    write_markdown_table(
        "table_5_kmeans_comparison.md",
        f"K-Means comparison on {kmeans_result['sample']} documents (k=20, same seed)",
        ["implementation", "fit time", "inertia", "NMI vs labels"],
        kmeans_rows,
    )

    bfs_rows: List[Tuple[object, ...]] = [
        (
            "custom BFS",
            f"{bfs_result['custom_ms']:.3f} ms",
            f"{bfs_result['custom_reached']} nodes",
        )
    ]
    networkx_result = bfs_result.get("networkx")
    if isinstance(networkx_result, dict):
        bfs_rows.append(
            (
                "networkx.bfs_edges",
                f"{networkx_result['ms']:.3f} ms",
                f"{networkx_result['reached']} nodes",
            )
        )
    write_markdown_table(
        "table_5_bfs_comparison.md",
        "BFS comparison on the real citation graph",
        ["implementation", "traversal time", "nodes reached"],
        bfs_rows,
    )

    results["figure"] = write_comparison_figure(results)
    return results


def write_comparison_figure(results: Dict[str, object]) -> str:
    """Draw the one-picture summary of every comparison in this benchmark.

    The brief asks each benchmark for a PNG beside its tables; the other four
    already write one, and this is benchmark 5's.  The chart is a signed log
    bar of *time ratio against the reference*, on a multiplicative axis, because
    the honest headline is that this project is slower than C in three places
    and faster than the two reference libraries where the algorithm differs.
    A linear axis would hide the 119x hash result behind the 1.18x K-Means one.

    Args:
        results: The aggregate mapping built by :func:`run`.

    Returns:
        The figure path.

    Complexity:
        Time O(1) beyond the numbers already computed; space O(1).
    """
    matplotlib_backend()
    import matplotlib.pyplot as plt

    hash_result = results["hash"]
    heap_result = results["heap"]
    bm25_result = results["bm25"]
    kmeans_result = results["kmeans"]
    bfs_result = results["bfs"]

    def ratio(mine: float, reference: float) -> float:
        """Mine / reference; < 1 means mine is faster."""
        return mine / max(1e-9, reference)

    labels: List[str] = []
    values: List[float] = []
    colours: List[str] = []

    labels.append("hash insert vs dict")
    values.append(ratio(hash_result["insert_ms"], hash_result["dict_build_ms"]))
    colours.append(PALETTE["naive"])

    labels.append("hash lookup vs dict")
    values.append(ratio(hash_result["lookup_ms"], hash_result["dict_lookup_ms"]))
    colours.append(PALETTE["naive"])

    labels.append("top-K heap vs heapq")
    values.append(ratio(heap_result["custom_ms"], heap_result["heapq_ms"]))
    colours.append(PALETTE["naive"])

    rank = bm25_result.get("rank_bm25")
    if isinstance(rank, dict):
        labels.append("BM25 vs rank_bm25")
        values.append(ratio(bm25_result["custom_ms_per_query"], rank["query_ms"]))
        colours.append(PALETTE["baseline"])

    sklearn_tfidf = bm25_result.get("sklearn_tfidf")
    if isinstance(sklearn_tfidf, dict):
        labels.append("BM25 vs sklearn TF-IDF")
        values.append(ratio(bm25_result["custom_ms_per_query"], sklearn_tfidf["query_ms"]))
        colours.append(PALETTE["baseline"])

    sklearn_kmeans = kmeans_result.get("sklearn")
    if isinstance(sklearn_kmeans, dict):
        labels.append("K-Means vs sklearn")
        values.append(ratio(kmeans_result["custom_ms"], sklearn_kmeans["ms"]))
        colours.append(PALETTE["ml"])

    networkx_result = bfs_result.get("networkx")
    if isinstance(networkx_result, dict):
        labels.append("BFS vs networkx")
        values.append(ratio(bfs_result["custom_ms"], networkx_result["ms"]))
        colours.append(PALETTE["graph"])

    figure, axes = plt.subplots(figsize=(9.0, 4.6))
    positions = list(range(len(values)))
    axes.barh(positions, values, color=colours, height=0.62)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8.5)
    axes.set_xscale("log")
    axes.axvline(1.0, color=PALETTE["grid"], linewidth=1.4, linestyle="--")
    axes.set_xlabel(
        "time relative to the reference library "
        "(log scale; left of the dashed line = mine is faster)"
    )
    axes.set_title("Benchmark 5: this implementation against the libraries", fontsize=10.5)
    for position, value in zip(positions, values):
        text = f"{value:.2f}x" if value >= 0.1 else f"{value:.3f}x"
        axes.annotate(
            text,
            xy=(value, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.0,
            color=PALETTE["reference"],
        )
    style_axes(axes)
    figure.tight_layout()
    path = figure_path("fig_5_comparison.png")
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


__all__ = [
    "run",
    "write_comparison_figure",
    "compare_hash_table",
    "compare_heap",
    "compare_bm25",
    "compare_kmeans",
    "compare_bfs",
]


def main() -> int:
    """Run the comparison on its own.

    Returns:
        Exit code.

    Complexity:
        Time proportional to the comparison workload (dominated by the
        ``rank_bm25`` and sklearn passes).
    """
    results = run()
    print(f"[compare_existing] summary keys: {sorted(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
