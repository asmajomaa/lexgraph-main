"""Benchmark 4: citation-graph structure and BFS scaling.

Reports the real graph's size, degree distribution and depth-vs-nodes-reached
table for the best-connected seed cases, then measures BFS time on synthetic
graphs of growing ``V + E`` so the ``O(V + E)`` claim is visible as a line.
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
from structures.graph import CitationGraph

#: Synthetic graph sizes ``(nodes, median_out_degree)``.
SYNTHETIC_SIZES: Tuple[Tuple[int, int], ...] = (
    (100, 2),
    (500, 2),
    (1000, 2),
    (2000, 2),
    (5000, 2),
    (10000, 2),
)


def build_synthetic_graph(nodes: int, degree: int, seed: int = 13) -> CitationGraph:
    """Build a random citation-like graph with a guaranteed connected spine.

    Args:
        nodes: Number of nodes.
        degree: Median out-degree.
        seed: RNG seed.

    Returns:
        A directed graph with ``nodes`` vertices.

    Complexity:
        Time O(nodes * degree); space O(nodes * degree).
    """
    rng = random.Random(seed)
    graph = CitationGraph()
    names = [f"n{index}" for index in range(nodes)]
    for name in names:
        graph.add_node(name)
    # A spine keeps the BFS from degenerating into isolated vertices.
    for index in range(nodes - 1):
        graph.add_edge(names[index], names[index + 1], "cited")
    for index in range(nodes):
        for _ in range(max(0, int(rng.gauss(degree, 0.7)))):
            graph.add_edge(names[index], names[rng.randrange(nodes)], "cited")
    return graph


def run() -> Dict[str, object]:
    """Execute benchmark 4 and write its figure and tables.

    Returns:
        Summary mapping.

    Complexity:
        Time O(max synthetic size x degree) for construction plus BFS per size;
        space O(largest graph).
    """
    from engine import get_engine

    matplotlib_backend()
    import matplotlib.pyplot as plt

    engine = get_engine(verbose=False)
    graph = engine.graph
    node_count = graph.node_count()
    edge_count = graph.edge_count()
    average_degree = graph.average_degree()
    print(
        f"[bench_graph] real graph: V={node_count} E={edge_count} "
        f"avg_degree={average_degree:.3f}"
    )

    edge_rows = [(edge_type, count) for edge_type, count in graph.edge_type_counts()]
    degree_histogram: Dict[int, int] = {}
    for node in graph.nodes():
        degree = graph.degree(node, "both")
        degree_histogram[degree] = degree_histogram.get(degree, 0) + 1

    # -- depth profile for the best-connected seeds ------------------------
    ranked = sorted(
        ((graph.degree(node, "both"), node) for node in graph.nodes()),
        reverse=True,
    )
    seed_cases = [node for _degree, node in ranked[:5]]
    depth_rows: List[Tuple[object, ...]] = []
    for seed_node in seed_cases:
        profile = graph.depth_profile(seed_node, max_depth=3, direction="both")
        title = engine.legal_index.titles[
            engine.legal_index.external_ids.index(seed_node)
        ] if seed_node in engine.legal_index.external_ids else seed_node
        depth_rows.append(
            (
                seed_node,
                str(title)[:60],
                graph.degree(seed_node, "out"),
                graph.degree(seed_node, "in"),
                " | ".join(f"d{depth}={reached}" for depth, reached in profile),
            )
        )
        print(f"[bench_graph] seed {seed_node}: {depth_rows[-1][-1]}")

    # -- BFS time vs graph size -------------------------------------------
    sizes: List[int] = []
    edge_totals: List[int] = []
    bfs_mean: List[float] = []
    bfs_std: List[float] = []
    for nodes, degree in SYNTHETIC_SIZES:
        synthetic = build_synthetic_graph(nodes, degree)
        sizes.append(synthetic.node_count())
        edge_totals.append(synthetic.edge_count())
        mean, std = time_call(
            lambda g=synthetic: g.bfs_traverse("n0", max_depth=10**9, direction="both"),
            repeats=5,
        )
        bfs_mean.append(mean * 1000.0)
        bfs_std.append(std * 1000.0)
        print(
            f"[bench_graph] synthetic V={synthetic.node_count():>6d} "
            f"E={synthetic.edge_count():>6d}  bfs={bfs_mean[-1]:8.3f} ms"
        )

    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    workload = [v + e for v, e in zip(sizes, edge_totals)]
    axes.errorbar(workload, bfs_mean, yerr=bfs_std, marker="o", linewidth=2,
                  color=PALETTE["graph"])
    axes.set_xlabel("graph size (V + E)")
    axes.set_ylabel("BFS time (ms, full traversal)")
    axes.set_title("BFS time vs graph size (O(V + E) shape)")
    style_axes(axes)
    figure.tight_layout()
    path = figure_path("fig_4_bfs_scaling.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)

    write_markdown_table(
        "table_4_graph_structure.md",
        "Legal citation graph structure",
        ["property", "value"],
        [
            ("nodes (cases)", node_count),
            ("edges (citations)", edge_count),
            ("average out-degree", f"{average_degree:.3f}"),
            ("isolated nodes (degree 0)", degree_histogram.get(0, 0)),
            (
                "nodes with degree >= 10",
                sum(count for d, count in degree_histogram.items() if d >= 10),
            ),
            ("distinct edge classes", len(edge_rows)),
            ("sources", engine.legal.source),
        ],
    )
    write_markdown_table(
        "table_4_edge_classes.md",
        "Citation edge classes (after resolving cited cases to the local slice)",
        ["edge class", "count"],
        edge_rows,
    )
    write_markdown_table(
        "table_4_depth_profiles.md",
        "BFS depth vs nodes reached for the five best-connected cases",
        ["seed", "case", "out-deg", "in-deg", "cumulative nodes by depth"],
        depth_rows,
    )
    write_csv(
        "table_4_bfs_scaling.csv",
        ["nodes", "edges", "bfs_mean_ms", "bfs_std_ms"],
        [
            (n, e, f"{m:.4f}", f"{s:.4f}")
            for n, e, m, s in zip(sizes, edge_totals, bfs_mean, bfs_std)
        ],
    )
    return {
        "nodes": node_count,
        "edges": edge_count,
        "average_degree": average_degree,
        "isolated": degree_histogram.get(0, 0),
        "edge_classes": edge_rows,
        "depth_profiles": depth_rows,
        "bfs_sizes": sizes,
        "bfs_edges": edge_totals,
        "bfs_ms": bfs_mean,
        "figure": path,
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


__all__ = ["run", "build_synthetic_graph", "SYNTHETIC_SIZES"]
