"""Benchmark 3: K-Means quality -- inertia curve, PCA scatter, purity/NMI/ARI.

Produces:

* ``fig_3a_kmeans_inertia.png``   -- inertia vs iteration,
* ``fig_3b_kmeans_scatter.png``   -- 2-D PCA projection coloured by cluster,
* ``fig_3c_kmeans_confusion.png`` -- cluster x true-label heatmap,
* ``table_3_kmeans_metrics.{csv,md}`` -- purity / NMI / ARI.

PCA is computed with numpy **in this benchmark only**; the clusterer itself is
pure Python, and the report says so.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from benchmarks.common import (
    PALETTE,
    figure_path,
    matplotlib_backend,
    style_axes,
    write_csv,
    write_markdown_table,
)
from config import KMEANS_SEED
from ml.evaluation import evaluate_clustering


def pca_2d(matrix: Sequence[Sequence[float]]) -> Tuple[List[float], List[float], List[float]]:
    """Project rows onto their first two principal components (numpy, benchmark only).

    Args:
        matrix: Dense ``n x d`` data.

    Returns:
        ``(x_coords, y_coords, explained_variance_ratio_first_two)``.

    Complexity:
        Time O(n * d^2) for the covariance eigen-decomposition (numpy, C-backed);
        space O(n * d) for the mean-centred copy.
    """
    import numpy as np

    data = np.asarray(matrix, dtype=float)
    data = data - data.mean(axis=0, keepdims=True)
    # Economy SVD is numerically safer than eigendecomposition of X^T X.
    _u, singular, vt = np.linalg.svd(data, full_matrices=False)
    projected = data @ vt[:2].T
    variance = (singular**2) / max(1, data.shape[0] - 1)
    total = float(variance.sum()) or 1.0
    ratio = [float(value / total) for value in variance[:2]]
    return projected[:, 0].tolist(), projected[:, 1].tolist(), ratio


def run() -> Dict[str, object]:
    """Execute benchmark 3 and write its figures and tables.

    Returns:
        Summary mapping with the metrics.

    Complexity:
        Time O(n * M) for the dense expansion plus O(n * M^2) for the SVD;
        space O(n * M) for the projection input.
    """
    from engine import get_engine

    matplotlib_backend()
    import matplotlib.pyplot as plt

    engine = get_engine(verbose=False)
    kmeans = engine.kmeans
    labels = list(kmeans.labels)
    truth = list(engine.news_index.labels)

    metrics = evaluate_clustering(truth, labels)
    print(
        f"[bench_kmeans] k={kmeans.k} iters={kmeans.n_iter} ({kmeans.converged}) "
        f"inertia={kmeans.inertia:.2f} purity={metrics['purity']:.4f} "
        f"nmi={metrics['nmi']:.4f} ari={metrics['ari']:.4f}"
    )

    # -- inertia curve -----------------------------------------------------
    figure, axes = plt.subplots(figsize=(6.4, 3.8))
    iterations = list(range(1, len(kmeans.inertia_history) + 1))
    axes.plot(iterations, kmeans.inertia_history, marker="o", markersize=3,
              linewidth=2, color=PALETTE["ml"], label="inertia")
    axes.set_xlabel("iteration")
    axes.set_ylabel("inertia (sum of squared distances)")
    # The first iteration dominates by an order of magnitude, so a linear axis
    # flattens the entire tail into one line.  The log axis shows both the
    # initial collapse and the slow final refinement.
    axes.set_yscale("log")
    axes.set_title(
        f"K-Means convergence (k={kmeans.k}, stopped: {kmeans.converged} after "
        f"{kmeans.n_iter} iters)"
    )
    # Annotate the marginal improvement of the last iteration, which is the
    # reason the max-iteration budget was exhausted.
    if len(kmeans.inertia_history) >= 2:
        tail_gain = kmeans.inertia_history[-2] - kmeans.inertia_history[-1]
        axes.annotate(
            f"last iteration improves inertia by {tail_gain:.2f}\n"
            f"(tol = {kmeans.tol:g})",
            xy=(iterations[-1], kmeans.inertia_history[-1]),
            xytext=(0.45, 0.22),
            textcoords="axes fraction",
            fontsize=8,
            color=PALETTE["reference"],
        )
    style_axes(axes)
    figure.tight_layout()
    inertia_path = figure_path("fig_3a_kmeans_inertia.png")
    figure.savefig(inertia_path, dpi=160)
    plt.close(figure)

    # -- PCA scatter -------------------------------------------------------
    vectors = engine.vectorizer.transform(engine.news.documents)
    dense = []
    for vector in vectors:
        row = [0.0] * vector.dimension
        for coordinate, value in zip(vector.indices, vector.values):
            row[coordinate] = value
        dense.append(row)
    xs, ys, ratio = pca_2d(dense)
    del dense
    del vectors

    # Two panels side by side: the first 5000 terms explain almost no variance
    # on their own (each is a rare word), so a single panel is uninformative.
    # Showing predicted clusters next to the true newsgroups makes the point
    # the report needs -- the clusterer recovers broad topical structure but
    # not the label boundaries -- instead of a hazy cloud.
    class_order = sorted(set(truth))
    class_index = {name: index for index, name in enumerate(class_order)}
    truth_codes = [class_index[name] for name in truth]

    figure, axes_list = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True, sharey=True)
    predicted = axes_list[0].scatter(
        xs, ys, c=labels, cmap="tab20", s=4, alpha=0.6, linewidths=0
    )
    axes_list[0].set_title("coloured by K-Means cluster")
    predicted_handles, _predicted_labels = predicted.legend_elements(num=min(kmeans.k, 20))
    axes_list[0].legend(
        predicted_handles,
        [f"c{index}" for index in range(len(predicted_handles))],
        title="cluster",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=5.5,
        frameon=False,
        ncol=1,
    )

    truth_plot = axes_list[1].scatter(
        xs, ys, c=truth_codes, cmap="tab20", s=4, alpha=0.6, linewidths=0
    )
    axes_list[1].set_title("coloured by true newsgroup")
    handles, _labels = truth_plot.legend_elements(num=min(len(class_order), 20))
    # Full names, not the last path segment: ``misc.forsale`` and
    # ``talk.politics.misc`` would both collapse to "misc" and the legend
    # would show one label twice.
    axes_list[1].legend(
        handles,
        class_order[: len(handles)],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=5.0,
        frameon=False,
        ncol=1,
        title="newsgroup",
    )

    for axes in axes_list:
        axes.set_xlabel(f"PC1 ({ratio[0] * 100:.2f}% variance)")
        style_axes(axes)
    axes_list[0].set_ylabel(f"PC2 ({ratio[1] * 100:.2f}% variance)")
    figure.suptitle(
        "20 Newsgroups projected with PCA: predicted clusters vs true labels",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 0.88, 0.94))
    scatter_path = figure_path("fig_3b_kmeans_scatter.png")
    figure.savefig(scatter_path, dpi=160)
    plt.close(figure)

    # -- confusion heatmap -------------------------------------------------
    from ml.evaluation import confusion_matrix

    classes, clusters, matrix = confusion_matrix(truth, labels)
    figure, axes = plt.subplots(figsize=(9.0, 6.4))
    image = axes.imshow(matrix, aspect="auto", cmap="YlGnBu")
    axes.set_xticks(range(len(clusters)))
    axes.set_xticklabels([str(c) for c in clusters], fontsize=7, rotation=90)
    axes.set_yticks(range(len(classes)))
    axes.set_yticklabels(classes, fontsize=7)
    axes.set_xlabel("predicted cluster")
    axes.set_ylabel("true newsgroup")
    axes.set_title("Cluster x true-label confusion matrix")
    figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    figure.tight_layout()
    confusion_path = figure_path("fig_3c_kmeans_confusion.png")
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)

    rows = [
        ("purity", f"{metrics['purity']:.4f}"),
        ("NMI", f"{metrics['nmi']:.4f}"),
        ("ARI", f"{metrics['ari']:.4f}"),
        ("documents", f"{int(metrics['n'])}"),
        ("clusters (k)", f"{int(metrics['n_pred'])}"),
        ("true classes", f"{int(metrics['n_true'])}"),
        ("iterations", f"{kmeans.n_iter}"),
        ("final inertia", f"{kmeans.inertia:.2f}"),
        ("largest cluster", f"{metrics['largest_cluster_fraction'] * 100:.1f}%"),
    ]
    write_csv("table_3_kmeans_metrics.csv", ["metric", "value"], rows)
    write_markdown_table(
        "table_3_kmeans_metrics.md",
        "K-Means clustering quality (k=20, seeded with k-means++)",
        ["metric", "value"],
        rows,
    )
    return {
        "metrics": metrics,
        "inertia_history": list(kmeans.inertia_history),
        "iterations": kmeans.n_iter,
        "converged": kmeans.converged,
        "inertia": kmeans.inertia,
        "figures": [inertia_path, scatter_path, confusion_path],
        "pca_variance": ratio,
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


__all__ = ["run", "pca_2d"]
