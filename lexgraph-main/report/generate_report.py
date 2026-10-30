"""Generate the API tables used by report section 4 and ``docs/complexity.md``.

The table is *derived from the code*, not typed by hand: every public function
is discovered with :mod:`ast`, its signature is reconstructed from the source,
and the ``Args`` / ``Returns`` / ``Complexity`` sections of its docstring are
parsed into table columns.  A function whose docstring lacks a complexity line
is reported loudly, so the table cannot silently drift from the code.

Writes:

* ``report/tables/table_7_public_api.md`` -- the full per-module table.
* ``docs/complexity.md``                   -- prose + the same table, grouped.

Usage::

    python -m report.generate_report
"""

from __future__ import annotations

import ast
import os
import re
from typing import Dict, List, Optional, Tuple

from config import REPORT_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT, "docs")
TABLES_DIR = os.path.join(str(REPORT_DIR), "tables")

#: Modules that form the graded core, in dependency order.  ``web/`` and
#: ``benchmarks/`` are deliberately excluded: they are allowed to use
#: ``dict``/``numpy`` and are not part of the engine's complexity story.
CORE_MODULES: List[str] = [
    "structures/hash_table.py",
    "structures/min_heap.py",
    "structures/graph.py",
    "structures/sparse_vector.py",
    "core/tokenizer.py",
    "core/inverted_index.py",
    "core/bm25.py",
    "ml/vectorizer.py",
    "ml/kmeans.py",
    "ml/evaluation.py",
    "core/agent.py",
]

_MODULE_TITLES: Dict[str, str] = {
    "structures/hash_table.py": "structures/hash_table.py -- hash table and postings",
    "structures/min_heap.py": "structures/min_heap.py -- binary heap and top-K",
    "structures/graph.py": "structures/graph.py -- citation graph and BFS",
    "structures/sparse_vector.py": "structures/sparse_vector.py -- sparse vectors",
    "core/tokenizer.py": "core/tokenizer.py -- tokenisation and stemming",
    "core/inverted_index.py": "core/inverted_index.py -- inverted index",
    "core/bm25.py": "core/bm25.py -- BM25 scoring",
    "ml/vectorizer.py": "ml/vectorizer.py -- TF-IDF vectoriser",
    "ml/kmeans.py": "ml/kmeans.py -- K-Means",
    "ml/evaluation.py": "ml/evaluation.py -- clustering metrics",
    "core/agent.py": "core/agent.py -- the ReAct agent",
}


def _clean(text: str) -> str:
    """Collapse a docstring fragment into one table cell.

    Args:
        text: Raw docstring text.

    Returns:
        Single-line, pipe-escaped text.

    Complexity: O(len(text)).
    """
    flat = re.sub(r"\s+", " ", text).strip()
    flat = flat.replace("|", "\\|")
    return flat


def _section(doc: str, name: str) -> str:
    """Pull one ``Args:``/``Returns:``/``Complexity:`` section out of a docstring.

    Args:
        doc: Full docstring.
        name: Section heading.

    Returns:
        The section body, or ``""``.

    Complexity: O(len(doc)).
    """
    match = re.search(
        rf"^{name}:[ \t]*([^\n]*)\n?((?:[ \t]+[^\n]*\n?)*)",
        doc,
        re.MULTILINE,
    )
    if not match:
        return ""
    inline, block = match.group(1), match.group(2)
    return _clean(inline + " " + block)


def _signature(node: ast.FunctionDef, source_lines: List[str]) -> str:
    """Reconstruct a readable signature from the source span.

    Args:
        node: Function node.
        source_lines: File lines.

    Returns:
        One-line signature such as ``search(self, query: str, k: int = 10)``.

    Complexity:
        Time O(number of signature lines); space O(len(signature)).
    """
    start = node.lineno - 1
    stop = node.body[0].lineno - 1 if node.body else start + 1
    header = " ".join(line.strip() for line in source_lines[start:stop])
    open_paren = header.find("(")
    depth = 0
    close_paren = -1
    for position in range(open_paren, len(header)):
        if header[position] == "(":
            depth += 1
        elif header[position] == ")":
            depth -= 1
            if depth == 0:
                close_paren = position
                break
    if open_paren < 0 or close_paren < 0:
        return node.name + "()"
    inner = _clean(header[open_paren + 1:close_paren])
    inner = re.sub(r"^self\s*(,\s*)?", "", inner)
    inner = re.sub(r",\s*$", "", inner)
    inner = re.sub(r"\s*,\s*", ", ", inner)
    return f"{node.name}({inner})"


def _returns(node: ast.FunctionDef, doc: str) -> str:
    """Extract the documented return value, or the annotation when absent.

    Args:
        node: Function node.
        doc: Docstring.

    Returns:
        Return description.

    Complexity: O(len(doc)).
    """
    described = _section(doc, "Returns")
    if described:
        return described
    if node.returns is not None:
        return _clean(ast.unparse(node.returns))
    return ""


def _complexity(doc: str) -> Tuple[str, str]:
    """Split the ``Complexity:`` section into time and space.

    Args:
        doc: Docstring.

    Returns:
        ``(time, space)`` strings.

    Complexity: O(len(doc)).
    """
    body = _section(doc, "Complexity")
    if not body:
        return "", ""
    time_match = re.search(r"\btime\b\s+([^;.]+)", body, re.IGNORECASE)
    space_match = re.search(r"\bspace\b\s+([^;.]+)", body, re.IGNORECASE)
    time = time_match.group(1).strip() if time_match else ""
    space = space_match.group(1).strip() if space_match else ""
    if not time and not space:
        # A bare ``Complexity: O(1).`` bounds both dimensions equally.
        if re.fullmatch(r"O\([^)]*\)\.?", body):
            return body.rstrip("."), body.rstrip(".")
        return body, ""
    return time, space


def collect(module: str) -> List[Dict[str, str]]:
    """Collect every public function of one module.

    Args:
        module: Path relative to the project root.

    Returns:
        Rows with ``name``, ``signature``, ``inputs``, ``output``, ``time``,
        ``space`` and ``summary`` keys.

    Complexity:
        Time O(size of the module); space O(rows).
    """
    path = os.path.join(ROOT, module)
    source = open(path, "r", encoding="utf-8").read()
    lines = source.splitlines()
    tree = ast.parse(source)
    rows: List[Dict[str, str]] = []

    def visit(node: ast.AST, owner: str = "") -> None:
        """Walk the tree, recording public methods and functions."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if child.name.startswith("_"):
                    continue
                visit(child, child.name)
            elif isinstance(child, ast.FunctionDef):
                if child.name.startswith("_"):
                    continue
                doc = ast.get_docstring(child) or ""
                time, space = _complexity(doc)
                first_line = doc.strip().split("\n\n")[0] if doc else ""
                rows.append(
                    {
                        "class": owner,
                        "name": child.name,
                        "signature": _signature(child, lines),
                        "inputs": _section(doc, "Args").replace("Args:", "").strip(),
                        "output": _returns(child, doc),
                        "time": time,
                        "space": space,
                        "summary": _clean(first_line),
                    }
                )

    visit(tree)
    return rows


def build_table() -> Tuple[str, List[str]]:
    """Render the markdown API table.

    Returns:
        ``(markdown, missing)`` where ``missing`` lists functions without a
        documented complexity.

    Complexity:
        Time O(total module size); space O(rows).
    """
    parts: List[str] = [
        "# Public API: input, output, time and space",
        "",
        "Generated from the source by `report/generate_report.py`; each row is a",
        "public function with the complexity stated in its docstring. Private",
        "helpers (`_bubble_up`, `_sink_down`, `_resize`, ...) are covered by",
        "`docs/complexity.md`.",
        "",
    ]
    missing: List[str] = []
    for module in CORE_MODULES:
        rows = collect(module)
        parts.append(f"## {_MODULE_TITLES.get(module, module)}")
        parts.append("")
        parts.append("| function | input | output | time | space |")
        parts.append("|---|---|---|---|---|")
        for row in rows:
            if not row["time"] or not row["space"]:
                missing.append(f"{module}:{row['class']}.{row['name']}")
            name = (
                f"`{row['class']}.{row['signature']}`"
                if row["class"]
                else f"`{row['signature']}`"
            )
            parts.append(
                "| {name} | {inputs} | {output} | {time} | {space} |".format(
                    name=name,
                    inputs=row["inputs"] or "-",
                    output=row["output"] or "-",
                    time=row["time"] or "**MISSING**",
                    space=row["space"] or "**MISSING**",
                )
            )
        parts.append("")
    return "\n".join(parts), missing


def main() -> int:
    """Write the API table and print a completeness check.

    Returns:
        Exit code (1 when a docstring lacks a complexity line).

    Complexity:
        Time O(total module size).
    """
    markdown, missing = build_table()
    table_path = os.path.join(TABLES_DIR, "table_7_public_api.md")
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    os.makedirs(DOCS_DIR, exist_ok=True)
    doc_path = os.path.join(DOCS_DIR, "complexity_api_table.md")
    with open(doc_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"[report] wrote {table_path}")
    print(f"[report] wrote {doc_path}")
    if missing:
        print(f"[report] {len(missing)} function(s) without a documented complexity:")
        for entry in missing:
            print(f"[report]   {entry}")
        return 1
    print("[report] every public function documents its time and space complexity")
    _ = Optional
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
