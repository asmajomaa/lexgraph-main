"""Benchmark 2: baseline vs ML-accelerated search, honestly measured.

For ``probe_clusters = 1, 2, 3`` this measures, over
:data:`config.EVAL_QUERY_COUNT` queries sampled from document titles:

* mean query time for each path (comparison baseline excluded from the ML
  path's own figure, reported separately),
* candidate documents scored and the fraction of the corpus that represents,
* **Recall@K** -- how many of the baseline's top-K documents the pruned search
  also returns -- which is the number that decides whether pruning is safe.

The run deliberately reports whatever the numbers say, including probe counts
where pruning loses recall and where it is slower than the baseline.
"""

from __future__ import annotations

import random
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
from config import BENCH_REPEATS, EVAL_QUERY_COUNT, KMEANS_SEED

#: Cluster counts to probe.
PROBE_SETTINGS: Tuple[int, ...] = (1, 2, 3)


def sample_queries(engine, count: int = EVAL_QUERY_COUNT, seed: int = KMEANS_SEED) -> List[str]:
    """Sample queries from document titles.

    Args:
        engine: A built engine.
        count: Number of queries.
        seed: RNG seed.

    Returns:
        Query strings of 2-4 content words, deduplicated.

    Complexity:
        Time O(count * title length); space O(count).
    """
    rng = random.Random(seed)
    tokenizer = engine.news_index.tokenizer
    seen: Dict[str, int] = {}
    queries: List[str] = []
    attempts = 0
    while len(queries) < count and attempts < count * 40:
        attempts += 1
        title = rng.choice(engine.news_index.titles)
        terms = tokenizer.tokenize(title)
        if len(terms) < 2:
            continue
        picked = terms[:3]
        query = " ".join(picked)
        if query in seen:
            continue
        seen[query] = 1
        queries.append(query)
    return queries


def run(repeats: int = BENCH_REPEATS) -> Dict[str, object]:
    """Execute benchmark 2 and write its figure and tables.

    Args:
        repeats: Unused for the aggregate run (each query is timed a fixed
            number of times below); kept for a uniform signature.

    Returns:
        Summary mapping with per-probe statistics.

    Complexity:
        Time O(probe_settings x queries x (baseline + pruned search));
        space O(corpus) for the reused score arrays.
    """
    from engine import get_engine

    matplotlib_backend()
    import matplotlib.pyplot as plt

    engine = get_engine(verbose=False)
    queries = sample_queries(engine)
    k = 10

    baseline_times: List[float] = []
    baseline_results: List[List[int]] = []
    baseline_scored: List[float] = []

    for query in queries:
        mean, _std = time_call(
            lambda: engine.agent.run(query, "baseline", k, compare_baseline=False),
            max(2, repeats),
        )
        baseline_times.append(mean * 1000.0)
        result = engine.agent.run(query, "baseline", k)
        baseline_results.append([int(row["doc_id"]) for row in result.results])
        baseline_scored.append(float(result.stats["candidates_scored"]))

    rows: List[Tuple[object, ...]] = []
    summary: Dict[str, object] = {"queries": len(queries), "probe": {}}

    for probe in PROBE_SETTINGS:
        ml_times: List[float] = []
        recalls: List[float] = []
        scored: List[float] = []
        pools: List[float] = []

        for index, query in enumerate(queries):
            mean, _std = time_call(
                lambda q=query, p=probe: engine.agent.run(q, "ml", k, probe_clusters=p),
                max(2, repeats),
            )
            result = engine.agent.run(query, "ml", k, probe_clusters=probe)
            ml_ids = {int(row["doc_id"]) for row in result.results}
            reference = set(baseline_results[index])
            recall = len(ml_ids & reference) / max(1, len(reference))
            ml_times.append(mean * 1000.0)
            recalls.append(recall)
            scored.append(float(result.stats["candidates_scored"]))
            pools.append(float(result.stats["candidate_pool"]))

        baseline_mean = sum(baseline_times) / len(baseline_times)
        ml_mean = sum(ml_times) / len(ml_times)
        corpus = float(engine.news_index.N)
        mean_scored = sum(scored) / len(scored)
        mean_pool = sum(pools) / len(pools)
        mean_recall = sum(recalls) / len(recalls)
        entry = {
            "baseline_ms": baseline_mean,
            "ml_ms": ml_mean,
            "speedup": baseline_mean / ml_mean if ml_mean else 0.0,
            "recall": mean_recall,
            "scored": mean_scored,
            "scored_fraction": mean_scored / corpus if corpus else 0.0,
            "pool": mean_pool,
            "pool_fraction": mean_pool / corpus if corpus else 0.0,
            "baseline_scored": sum(baseline_scored) / len(baseline_scored),
        }
        summary["probe"][probe] = entry
        rows.append(
            (
                probe,
                f"{baseline_mean:.3f}",
                f"{ml_mean:.3f}",
                f"{baseline_mean / ml_mean:.2f}",
                f"{mean_recall * 100:.1f}%",
                f"{mean_pool:.0f}",
                f"{mean_pool / corpus * 100:.1f}%",
                f"{mean_scored:.0f}",
                f"{mean_scored / corpus * 100:.1f}%",
            )
        )
        print(
            f"[bench_query_time_ml] probe={probe}  baseline={baseline_mean:6.3f} ms  "
            f"ml={ml_mean:6.3f} ms  speedup={baseline_mean / ml_mean:5.2f}x  "
            f"recall@10={mean_recall * 100:5.1f}%  pool={mean_pool:.0f} "
            f"({mean_pool / corpus * 100:.1f}% of corpus)"
        )

    probes = list(PROBE_SETTINGS)
    speedups = [summary["probe"][p]["speedup"] for p in probes]
    recalls = [summary["probe"][p]["recall"] * 100 for p in probes]
    scored_pct = [summary["probe"][p]["scored_fraction"] * 100 for p in probes]
    pool_pct = [summary["probe"][p]["pool_fraction"] * 100 for p in probes]

    figure, axes_list = plt.subplots(1, 3, figsize=(12.6, 3.9))
    labels = [f"probe={p}" for p in probes]

    axes_list[0].bar(labels, speedups, color=PALETTE["ml"])
    axes_list[0].axhline(1.0, color=PALETTE["reference"], linestyle="--", linewidth=1)
    axes_list[0].set_title("Speedup over baseline")
    axes_list[0].set_ylabel("x (baseline / ML)")

    axes_list[1].bar(labels, recalls, color=PALETTE["baseline"])
    axes_list[1].set_ylim(0, 105)
    axes_list[1].set_title("Recall@10 vs baseline")
    axes_list[1].set_ylabel("%")

    width = 0.36
    positions = list(range(len(probes)))
    axes_list[2].bar(
        [p - width / 2 for p in positions],
        pool_pct,
        width,
        color=PALETTE["ml"],
        label="candidate pool",
    )
    axes_list[2].bar(
        [p + width / 2 for p in positions],
        scored_pct,
        width,
        color=PALETTE["index"],
        label="scored",
    )
    axes_list[2].set_xticks(positions)
    axes_list[2].set_xticklabels(labels)
    axes_list[2].set_title("Corpus touched")
    axes_list[2].set_ylabel("% of corpus")
    axes_list[2].legend(frameon=False, fontsize=8)

    for axes in axes_list:
        style_axes(axes)
    figure.suptitle("Baseline vs K-Means-accelerated BM25", fontsize=12)
    figure.tight_layout()
    path = figure_path("fig_2_baseline_vs_ml.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)

    write_csv(
        "table_2_baseline_vs_ml.csv",
        [
            "probe_clusters", "baseline_ms", "ml_ms", "speedup",
            "recall_at_10", "pool", "pool_fraction", "scored",
            "scored_fraction",
        ],
        rows,
    )
    write_markdown_table(
        "table_2_baseline_vs_ml.md",
        "Baseline vs ML-accelerated BM25 (100 sampled queries, k=10)",
        [
            "probe", "baseline ms", "ML ms", "speedup", "recall@10",
            "pool", "pool %", "scored", "scored %",
        ],
        rows,
    )
    summary["figure"] = path
    return summary


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


__all__ = ["run", "sample_queries", "PROBE_SETTINGS"]
