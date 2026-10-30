"""Style tests: the PEP 8 line-length agreement must be visible and enforced.

PEP 8 fixes 79 characters as the default maximum but explicitly allows a team
to raise it to 99 "provided that they agree on it" and document the decision.
``pyproject.toml`` records the agreement; this module is what makes it
checkable, so the report's PEP 8 claim rests on a test rather than on prose.
"""

from __future__ import annotations

import os
from typing import List

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Directories checked: the graded engine plus the loaders and front ends.
#: Tests and benchmarks are included too, so the whole repo reads the same way.
CHECKED_DIRS = ("structures", "core", "ml", "data", "benchmarks", "web", "tests")

#: The documented maximum, matching ``[tool.black] line-length``.
LINE_LIMIT = 99


def _python_files() -> List[str]:
    """Every checked Python file.

    Returns:
        Absolute paths, sorted.

    Complexity:
        Time O(number of files); space O(number of files).
    """
    paths: List[str] = []
    for directory in CHECKED_DIRS:
        root = os.path.join(_PROJECT_ROOT, directory)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.endswith(".py"):
                paths.append(os.path.join(root, name))
    return paths


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: os.path.basename(p))
def test_lines_are_within_the_declared_limit(path: str) -> None:
    """No line may exceed the project's documented 99-character limit."""
    with open(path, "r", encoding="utf-8") as handle:
        offending = [
            (number, len(line.rstrip("\n")), line.rstrip("\n")[:70])
            for number, line in enumerate(handle, 1)
            if len(line.rstrip("\n")) > LINE_LIMIT
        ]
    assert not offending, (
        f"{os.path.basename(path)} has {len(offending)} line(s) over "
        f"{LINE_LIMIT} characters:\n"
        + "\n".join(f"  {n}:{length} {text}" for length, n, text in
                    [(n, length, text) for n, length, text in offending])
    )


def test_the_limit_is_declared_in_pyproject() -> None:
    """The relaxed limit must be written down, not merely used."""
    config = os.path.join(_PROJECT_ROOT, "pyproject.toml")
    assert os.path.exists(config), "pyproject.toml declares the line-length agreement"
    with open(config, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert f"line-length = {LINE_LIMIT}" in text
