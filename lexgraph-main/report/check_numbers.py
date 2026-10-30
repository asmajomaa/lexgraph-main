"""Check the report's quoted numbers against the committed benchmark outputs.

This is an *auditor*, not a rewriter. Rewriting thirty documents
automatically is how prose gets mangled; the safer contract is: every claim
that carries a number is registered here with the value the current run
produced, and this script fails loudly when a document disagrees.

Usage::

    python -m report.check_numbers          # report mismatches, exit 1
    python -m report.check_numbers --list   # show every registered claim

Each entry is ``(file, exact text in the file, label, value from the run)``.
The text is matched literally, so a claim that has been edited without updating
its number shows up as a miss rather than silently passing.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "report", "tables")


def load_summary() -> Dict[str, object]:
    """Read ``report/tables/summary.json``.

    Returns:
        The parsed mapping.

    Complexity: Time O(size of file); space O(size of file).
    """
    path = os.path.join(TABLES, "summary.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(name: str) -> List[Dict[str, str]]:
    """Read one benchmark CSV.

    Args:
        name: File name inside ``report/tables``.

    Returns:
        Rows keyed by header.

    Complexity: Time O(rows); space O(rows).
    """
    path = os.path.join(TABLES, name)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read(path: str) -> str:
    """Read a project file.

    Args:
        path: Path relative to the project root.

    Returns:
        The text.

    Complexity: Time O(size of file); space O(size of file).
    """
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as handle:
        return handle.read()


#: Registered claims: ``(file, regex, label, value provider)``.
#: The regex must contain one capture group holding the number as printed.
CLAIMS: List[Tuple[str, str, str]] = [
    ("report/report_sections/section_8.md",
     r"\| insert 150,000 keys \| ([\d,\.]+) ms \|", "hash insert ms"),
    ("report/report_sections/section_8.md",
     r"\| insert 150,000 keys \|[^|]+\|[^|]+\| \*\*([\d\.]+)× slower\*\* \|",
     "hash insert ratio"),
    ("report/report_sections/section_8.md",
     r"\| lookup 150,000 keys \| ([\d,\.]+) ms \|", "hash lookup ms"),
    ("report/report_sections/section_8.md",
     r"\| peak memory \| ([\d,]+) KiB \|", "hash peak KiB"),
    ("report/report_sections/section_8.md",
     r"\| time \| ([\d\.]+) ms \| ([\d\.]+) ms \|", "heap ms pair"),
    ("report/report_sections/section_8.md",
     r"\| \*\*custom BM25\*\* \| index built at engine build \| \*\*([\d\.]+) ms\*\* \|",
     "bm25 custom ms"),
    ("report/report_sections/section_7.md",
     r"\| 18,719 \| ([\d\.]+) ± [\d\.]+ \|", "scalability custom ms at full corpus"),
    ("report/report_sections/section_7.md",
     r"\| 18,719 \|[^|]+ \| ([\d,\.]+) ± [\d\.]+ \|",
     "scalability naive ms at full corpus"),
    ("report/report_sections/section_7.md",
     r"\| 1 \| [\d\.]+ \| [\d\.]+ \| \*\*([\d\.]+)×\*\* \|", "probe-1 speedup"),
    ("report/report_sections/section_7.md",
     r"\| 2 \| [\d\.]+ \| [\d\.]+ \| \*\*([\d\.]+)×\*\* \|", "probe-2 speedup"),
]


def expected() -> Dict[str, List[float]]:
    """Compute the values the registered claims should hold.

    Returns:
        Label -> list of expected numbers in capture order.

    Complexity: Time O(1) beyond loading files; space O(claims).
    """
    summary = load_summary()
    compare = summary["compare"]
    hash_result = compare["hash"]
    heap_result = compare["heap"]
    bm25_result = compare["bm25"]
    scalability = load_csv("table_1_query_time.csv")
    pruning = load_csv("table_2_baseline_vs_ml.csv")
    last = scalability[-1]

    return {
        "hash insert ms": [float(hash_result["insert_ms"])],
        "hash insert ratio": [
            float(hash_result["insert_ms"]) / max(1e-9, float(hash_result["dict_build_ms"]))
        ],
        "hash lookup ms": [float(hash_result["lookup_ms"])],
        "hash peak KiB": [float(hash_result["peak_kib"])],
        "heap ms pair": [
            float(heap_result["custom_ms"]),
            float(heap_result["heapq_ms"]),
        ],
        "bm25 custom ms": [float(bm25_result["custom_ms_per_query"])],
        "scalability custom ms at full corpus": [float(last["custom_mean_ms"])],
        "scalability naive ms at full corpus": [float(last["naive_mean_ms"])],
        "probe-1 speedup": [float(pruning[0]["speedup"])],
        "probe-2 speedup": [float(pruning[1]["speedup"])],
    }


def _to_float(text: str) -> float:
    """Parse a printed number, tolerating thousands separators.

    Args:
        text: The captured text.

    Returns:
        The float value.

    Complexity: Time O(len(text)); space O(1).
    """
    return float(text.replace(",", ""))


def check(tolerance: float = 0.011) -> List[str]:
    """Compare every registered claim with the run.

    Args:
        tolerance: Allowed relative difference, covering print rounding.

    Returns:
        Human-readable mismatch descriptions.

    Complexity: Time O(claims x file size); space O(mismatches).
    """
    wanted = expected()
    problems: List[str] = []
    cache: Dict[str, str] = {}
    for path, pattern, label in CLAIMS:
        if path not in cache:
            cache[path] = read(path)
        text = cache[path]
        match = re.search(pattern, text)
        if match is None:
            problems.append(f"{path}: claim {label!r} not found (pattern changed?)")
            continue
        found = [_to_float(group) for group in match.groups()
                 if group and re.fullmatch(r"[\d,\.]+", group)]
        for got, want in zip(found, wanted[label]):
            allowed = max(abs(want) * tolerance, 0.01)
            if abs(got - want) > allowed:
                problems.append(
                    f"{path}: {label} says {got} but the run produced {want:.4f}"
                )
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    """Run the check.

    Args:
        argv: Arguments (``--list`` prints the registry).

    Returns:
        Exit code: 0 when every claim matches.

    Complexity: Time O(claims); space O(problems).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--list" in args:
        for path, _pattern, label in CLAIMS:
            print(f"{label:44s} {path}")
        return 0
    problems = check()
    if problems:
        print(f"[numbers] {len(problems)} mismatch(es):")
        for problem in problems:
            print("  " + problem)
        return 1
    print(f"[numbers] {len(CLAIMS)} registered claims agree with the committed run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
