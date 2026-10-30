"""Benchmark 1: query time vs corpus size -- custom index vs naive linear scan.

For each size in ``1k, 2k, 5k, 10k, 18k`` the benchmark builds a nested prefix
of the newsgroups corpus, then times the same query set twice:

* **custom** -- tokenise, hash-table lookup, BM25 over postings, heap top-K;
* **naive** -- tokenise every document, count matching terms, score with the
  same BM25 formula, sort the full list.

The naive scan is the honest baseline for "why an inverted index at all": it
does the same arithmetic with the same tokenizer, so the ratio is purely the
index's contribution.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence, Tuple

from benchmarks.common import (
    PALETTE,
    figure_path,
    matplotlib_backend,
    style_axes,
    time_call,
    write_csv,
    write_markdown_table,
)
from config import BENCH_REPEATS
from core.bm25 import BM25
from core.inverted_index import InvertedIndex
from core.tokenizer import Tokenizer

#: Corpus sizes to measure.  18 719 is the full bydate corpus.
SIZES: Tuple[int, ...] = (1000, 2000, 5000, 10000, 18719)

#: Queries held fixed across sizes so the comparison is like-for-like.
QUERIES: Tuple[str, ...] = (
    "hockey playoffs goalie",
    "graphics texture rendering",
    "space shuttle orbit nasa",
    "government tax policy",
    "religion faith church",
)


def naive_search(
    documents: Sequence[str],
    query: str,
    tokenizer: Tokenizer,
    k: int = 10,
) -> List[Tuple[float, int]]:
    """Linear scan with no index: tokenise every document on every query.

    Args:
        documents: Raw document texts.
        query: Raw query.
        tokenizer: Shared tokenizer.
        k: Number of results.

    Returns:
        ``(score, doc_id)`` pairs, best first.

    Complexity:
        Time O(N * (len(doc) + T_q)) -- the tokenisation dominates; space O(N)
        for the scores plus O(vocab) per document.
    """
    query_terms = tokenizer.tokenize(query)
    if not query_terms:
        return []
    unique_terms = []
    seen: Dict[str, int] = {}
    for term in query_terms:
        if term not in seen:
            seen[term] = 1
            unique_terms.append(term)

    tokenized: List[Dict[str, int]] = []
    lengths: List[int] = []
    document_frequency: Dict[str, int] = {}
    for text in documents:
        counts: Dict[str, int] = {}
        for term in tokenizer.tokenize(text):
            counts[term] = counts.get(term, 0) + 1
        tokenized.append(counts)
        lengths.append(sum(counts.values()))
        for term in unique_terms:
            if term in counts:
                document_frequency[term] = document_frequency.get(term, 0) + 1

    n_docs = len(documents)
    average_length = sum(lengths) / n_docs if n_docs else 1.0
    k1, b = 1.5, 0.75
    import math

    idf = {
        term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }

    scored: List[Tuple[float, int]] = []
    for doc_id, counts in enumerate(tokenized):
        total = 0.0
        for term in unique_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denominator = tf + k1 * (1.0 - b + b * lengths[doc_id] / average_length)
            total += idf[term] * tf * (k1 + 1.0) / denominator
        if total > 0.0:
            scored.append((total, doc_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored[:k]


def run(repeats: int = BENCH_REPEATS) -> Dict[str, object]:
    """Execute benchmark 1 and write its figure and tables.

    Args:
        repeats: Measurements per size per method.

    Returns:
        Summary mapping used by ``run_all`` and the report.

    Complexity:
        Time O(sum over sizes of N * tokenisation) for the naive side; the
        custom side is milliseconds.  Space O(largest index).
    """
    from engine import get_engine

    matplotlib_backend()
    import matplotlib.pyplot as plt

    engine = get_engine(verbose=False)
    documents = engine.news.documents
    tokenizer = Tokenizer()

    custom_mean: List[float] = []
    custom_std: List[float] = []
    naive_mean: List[float] = []
    naive_std: List[float] = []

    for size in SIZES:
        subset = documents[:size]
        index = InvertedIndex.build(subset, tokenizer, name=f"bench-{size}")
        scorer = BM25(index)

        def run_custom() -> None:
            """Time the indexed search over the fixed query set."""
            for query in QUERIES:
                scorer.search(query, 10)

        def run_naive() -> None:
            """Time the linear scan over the fixed query set."""
            texts = [document.text for document in subset]
            for query in QUERIES:
                naive_search(texts, query, tokenizer, 10)

        mean, std = time_call(run_custom, repeats)
        custom_mean.append(mean / len(QUERIES) * 1000.0)
        custom_std.append(std / len(QUERIES) * 1000.0)

        mean, std = time_call(run_naive, max(1, repeats // 2))
        naive_mean.append(mean / len(QUERIES) * 1000.0)
        naive_std.append(std / len(QUERIES) * 1000.0)

        print(
            f"[bench_query_time] N={size:>6d}  custom={custom_mean[-1]:8.3f} ms  "
            f"naive={naive_mean[-1]:10.3f} ms  ratio={naive_mean[-1] / custom_mean[-1]:7.1f}x"
        )

    figure = plt.figure(figsize=(7.2, 4.2))
    axes = figure.add_subplot(1, 1, 1)
    axes.errorbar(
        SIZES, custom_mean, yerr=custom_std, marker="o", linewidth=2,
        color=PALETTE["index"], label="custom inverted index + BM25",
    )
    axes.errorbar(
        SIZES, naive_mean, yerr=naive_std, marker="s", linewidth=2,
        color=PALETTE["naive"], label="naive linear scan",
    )
    axes.set_xlabel("corpus size (documents)")
    axes.set_ylabel("mean query time (ms)")
    axes.set_title("Query time vs corpus size")
    axes.set_yscale("log")
    axes.legend(frameon=False)
    style_axes(axes)
    figure.tight_layout()
    path = figure_path("fig_1_query_time_vs_size.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)

    write_csv(
        "table_1_query_time.csv",
        [
            "corpus_size", "custom_mean_ms", "custom_std_ms",
            "naive_mean_ms", "naive_std_ms", "speedup_x",
        ],
        [
            (size, f"{cm:.4f}", f"{cs:.4f}", f"{nm:.4f}", f"{ns:.4f}", f"{nm / cm:.2f}")
            for size, cm, cs, nm, ns in zip(SIZES, custom_mean, custom_std, naive_mean, naive_std)
        ],
    )
    write_markdown_table(
        "table_1_query_time.md",
        "Query time vs corpus size (mean of 5 fixed queries)",
        ["corpus", "custom (ms)", "naive (ms)", "speedup"],
        [
            (size, f"{cm:.3f}", f"{nm:.3f}", f"{nm / cm:.1f}x")
            for size, cm, nm in zip(SIZES, custom_mean, naive_mean)
        ],
    )
    return {
        "sizes": list(SIZES),
        "custom_mean_ms": custom_mean,
        "naive_mean_ms": naive_mean,
        "figure": path,
        "max_speedup": max(
            nm / cm for cm, nm in zip(custom_mean, naive_mean)
        ),
    }


def main() -> int:
    """Run this benchmark on its own.

    Returns:
        Exit code.

    Complexity:
        Time proportional to the benchmark's own workload.
    """
    result = run()
    print(f"[{__spec__.name.split('.')[-1]}] summary keys: {sorted(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run", "naive_search", "SIZES", "QUERIES"]
