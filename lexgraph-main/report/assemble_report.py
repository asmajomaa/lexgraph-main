"""Assemble the nine report sections into one document.

``report/report_sections/section_N.md`` stays the source of truth (each file is
independently reviewable); this script concatenates them with a title page and a
table of contents into ``report/report.md``. Running it after editing a section
keeps the combined document in step.

Usage::

    python -m report.assemble_report
"""

from __future__ import annotations

import os
from typing import List

from config import REPORT_DIR

SECTIONS_DIR = os.path.join(str(REPORT_DIR), "report_sections")

TITLE_PAGE = """# LEXGRAPH-RAG: A Hybrid Agentic Information Retrieval Engine

**University project — ISIMM, University of Monastir**

| | |
|---|---|
| Course | Information Retrieval / Machine Learning / Graph Theory |
| Subject | Sujet 6 — LEXGRAPH-RAG |
| Datasets | 20 Newsgroups (Mitchell, 1997); Legal Case Reports (Galgani, 2010) |
| Engine | pure Python; no `dict`-as-index, `heapq`, `networkx`, `scikit-learn` or `numpy` in the core |
| Evidence | every number in this report was produced by running the committed code |

> **Data provenance.** All retrieval, clustering and graph results below were
> produced from the **real** datasets (18,719 newsgroup posts; 3,890 legal
> cases). The synthetic generators exist only so the project runs on a machine
> without the corpora, and no reported result comes from them.
>
> **Reproduction.** `python main.py --build`, then `python -m benchmarks.run_all`
> regenerates every figure and table quoted here; `pytest` runs the 164 tests.

"""

TOC = """## Contents

1. [Objective of the project](#1-objective-of-the-project)
2. [The problem solved by the algorithm](#2-the-problem-solved-by-the-algorithm)
3. [The algorithm](#3-the-algorithm)
4. [Algorithm and data-structure analysis](#4-algorithm-and-data-structure-analysis)
5. [Implementation of the data structures](#5-implementation-of-the-data-structures)
6. [Implementation of the algorithm](#6-implementation-of-the-algorithm)
7. [Experimental results](#7-experimental-results)
8. [Comparison with existing implementations](#8-comparison-with-existing-implementations)
9. [Conclusion](#9-conclusion)

Appendices: [complexity notes](../docs/complexity.md) ·
[design decisions](../docs/design_decisions.md) ·
[per-function API table](tables/table_7_public_api.md) ·
[defense script](defense_script.md)

---

"""


def assemble() -> str:
    """Concatenate the nine sections.

    Returns:
        The combined markdown.

    Complexity:
        Time O(total size of the sections); space O(same).
    """
    parts: List[str] = [TITLE_PAGE, TOC]
    for number in range(1, 10):
        path = os.path.join(SECTIONS_DIR, f"section_{number}.md")
        if not os.path.exists(path):
            parts.append(f"# {number}. (missing section file)\n\n")
            continue
        with open(path, "r", encoding="utf-8") as handle:
            parts.append(handle.read().rstrip() + "\n\n---\n\n")
    return "".join(parts)


def main() -> int:
    """Write ``report/report.md``.

    Returns:
        Exit code.

    Complexity:
        Time O(total size of the sections).
    """
    text = assemble()
    path = os.path.join(str(REPORT_DIR), "report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    words = len(text.split())
    print(f"[report] wrote {path} ({words} words, {len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
