"""Regenerate every figure and table in the report.

Usage::

    python -m benchmarks.run_all
    python -m benchmarks.run_all --only kmeans graph     # a subset
    python -m benchmarks.run_all --skip compare          # skip the slowest one

Each benchmark writes a JSON summary into ``report/tables/summary.json`` as it
finishes, so ``report/generate_report.py`` can quote measured numbers without
re-running thousands of queries.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

from config import TABLES_DIR

#: Benchmark name -> (module path, human label).  Order matters: the cheap
#: structural benchmarks run first so a long run fails late rather than early.
BENCHMARKS: List[tuple[str, str, str]] = [
    ("graph", "benchmarks.bench_graph", "graph structure + BFS scaling"),
    ("kmeans", "benchmarks.bench_kmeans", "K-Means quality + PCA"),
    ("scalability", "benchmarks.bench_scalability", "query time vs corpus size"),
    ("query_time", "benchmarks.bench_query_time", "baseline vs ML-accelerated"),
    ("compare", "benchmarks.compare_existing", "comparison with existing libraries"),
]


def _summary_path() -> str:
    """Location of the JSON summary.

    Returns:
        Path string.

    Complexity: O(1).
    """
    return os.path.join(str(TABLES_DIR), "summary.json")


def load_summary() -> Dict[str, object]:
    """Read the accumulated summary, or an empty mapping.

    Returns:
        Summary mapping.

    Complexity:
        Time O(file size).
    """
    path = _summary_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_summary(summary: Dict[str, object]) -> str:
    """Persist the summary as JSON.

    Args:
        summary: Mapping to write.

    Returns:
        Path written.

    Complexity:
        Time O(size of summary).
    """
    os.makedirs(str(TABLES_DIR), exist_ok=True)
    path = _summary_path()

    def default(value):  # type: ignore[no-untyped-def]
        """Coerce objects JSON cannot handle."""
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=default)
    return path


def run_selected(names: Optional[List[str]] = None) -> Dict[str, object]:
    """Run the selected benchmarks.

    Args:
        names: Benchmark names; ``None`` runs all.

    Returns:
        The accumulated summary.

    Complexity:
        Time proportional to the selected benchmarks; space O(summary).
    """
    from benchmarks.common import matplotlib_backend

    matplotlib_backend()
    import importlib

    summary = load_summary()
    selected = [entry for entry in BENCHMARKS if names is None or entry[0] in names]
    if not selected:
        raise SystemExit(f"unknown benchmark names: {names}")

    for name, module_path, label in selected:
        started = time.perf_counter()
        print(f"\n=== [{name}] {label} ===")
        module = importlib.import_module(module_path)
        result = module.run()
        elapsed = time.perf_counter() - started
        summary[name] = result
        summary.setdefault("timings", {})[name] = round(elapsed, 2)
        save_summary(summary)
        print(f"=== [{name}] done in {elapsed:.1f}s ===")

    path = save_summary(summary)
    print(f"\nsummary -> {path}")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments.

    Returns:
        Exit code.

    Complexity:
        Time proportional to the selected benchmarks.
    """
    parser = argparse.ArgumentParser(description="Run LEXGRAPH-RAG benchmarks")
    parser.add_argument("--only", nargs="+", help="run only these benchmarks")
    parser.add_argument("--skip", nargs="+", default=[], help="skip these benchmarks")
    args = parser.parse_args(argv)
    names: Optional[List[str]] = None
    if args.only:
        names = list(args.only)
    for skip in args.skip:
        if names is None:
            names = [entry[0] for entry in BENCHMARKS]
        if skip in names:
            names.remove(skip)
    run_selected(names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
