"""Shared benchmark utilities: timing, CSV/markdown output, figure setup.

Every benchmark reports ``mean +- std`` over repeated measurements taken with
``time.perf_counter``, and each one writes both a PNG and a CSV/markdown table
so the report can cite exact numbers rather than transcribed screenshots.
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from config import BENCH_REPEATS, FIGURES_DIR, TABLES_DIR


def _ensure_dirs() -> None:
    """Create the figure and table directories if missing.

    Complexity: O(1).
    """
    os.makedirs(str(FIGURES_DIR), exist_ok=True)
    os.makedirs(str(TABLES_DIR), exist_ok=True)


def time_call(function: Callable[[], object], repeats: int = BENCH_REPEATS) -> Tuple[float, float]:
    """Time ``function`` repeatedly and report mean and standard deviation.

    Args:
        function: Zero-argument callable.
        repeats: Number of measurements.

    Returns:
        ``(mean_seconds, std_seconds)``.

    Complexity:
        Time ``repeats`` times the callable's cost; space O(repeats).
    """
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    if len(samples) == 1:
        return samples[0], 0.0
    return statistics.mean(samples), statistics.pstdev(samples)


def time_setup_then_call(
    setup: Callable[[], object],
    function: Callable[[object], object],
    repeats: int = BENCH_REPEATS,
) -> Tuple[float, float, object]:
    """Time ``function(setup())`` with setup repeated for every measurement.

    Use this when the work being measured consumes state (a drained heap, a
    filled bitmap) and must be rebuilt between repetitions.

    Args:
        setup: Produces the input.
        function: Consumes it.
        repeats: Number of measurements.

    Returns:
        ``(mean_seconds, std_seconds, last_setup_result)``.

    Complexity:
        Time O(repeats x (setup + function)); space O(1) beyond the last input.
    """
    samples: List[float] = []
    last: object = None
    for _ in range(max(1, repeats)):
        last = setup()
        started = time.perf_counter()
        function(last)
        samples.append(time.perf_counter() - started)
    if len(samples) == 1:
        return samples[0], 0.0, last
    return statistics.mean(samples), statistics.pstdev(samples), last


def write_csv(name: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Write a CSV table under ``report/tables``.

    Args:
        name: File name including ``.csv``.
        header: Column names.
        rows: Data rows.

    Returns:
        The path written.

    Complexity:
        Time O(rows x columns); space O(1) beyond the rows.
    """
    _ensure_dirs()
    path = os.path.join(str(TABLES_DIR), name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return path


def write_markdown_table(
    name: str,
    title: str,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    """Write a Markdown table under ``report/tables``.

    Args:
        name: File name including ``.md``.
        title: Heading placed above the table.
        header: Column names.
        rows: Data rows.

    Returns:
        The path written.

    Complexity:
        Time O(rows x columns); space O(rows x columns).
    """
    _ensure_dirs()
    path = os.path.join(str(TABLES_DIR), name)
    lines = [f"# {title}", "", "| " + " | ".join(str(col) for col in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def figure_path(name: str) -> str:
    """Absolute path for a figure file.

    Args:
        name: File name including ``.png``.

    Returns:
        Path string.

    Complexity: O(1).
    """
    _ensure_dirs()
    return os.path.join(str(FIGURES_DIR), name)


def matplotlib_backend() -> str:
    """Select a headless matplotlib backend and return its name.

    Returns:
        The backend that was selected.

    Complexity: O(1).
    """
    import matplotlib

    matplotlib.use("Agg")
    return matplotlib.get_backend()


#: A restrained palette shared by every figure so the report reads as one set.
PALETTE: Dict[str, str] = {
    "baseline": "#0f6b6b",
    "ml": "#b7791f",
    "index": "#4b4acf",
    "naive": "#a03a3a",
    "graph": "#7a4b9a",
    "reference": "#5d6b7a",
    "grid": "#d5dbe2",
}


def style_axes(axes) -> None:  # type: ignore[no-untyped-def]
    """Apply the shared minimalist style to an Axes.

    Args:
        axes: A matplotlib Axes.

    Complexity: O(1).
    """
    axes.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.8)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def summarize(values: Sequence[float]) -> str:
    """Format a value sequence as ``mean +- std``.

    Args:
        values: Measurements.

    Returns:
        Human-readable summary with 4 significant digits.

    Complexity:
        Time O(n); space O(1).
    """
    if not values:
        return "n/a"
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +- {std:.4f}"


__all__ = [
    "time_call",
    "time_setup_then_call",
    "write_csv",
    "write_markdown_table",
    "figure_path",
    "matplotlib_backend",
    "style_axes",
    "summarize",
    "PALETTE",
]
