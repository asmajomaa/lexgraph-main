"""Run the test suite and write structured results for the workbench.

Usage::

    python -m web.run_tests

Writes ``report/tables/tests.json`` with per-test outcomes so the Tests panel
can render the real suite instead of a hard-coded list.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List

from config import PACKAGE_ROOT, TABLES_DIR

#: Matches pytest's verbose line, e.g.
#: ``tests/test_x.py::test_y[param with spaces] PASSED [ 12%]``.
#: The test id is greedy up to the status word, because parametrised ids
#: legitimately contain spaces and brackets.
_RESULT_LINE = re.compile(
    r"^(?P<file>\S+?\.py)::(?P<name>.+?)\s+(?P<status>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b"
)
_SUMMARY_LINE = re.compile(r"^(?P<count>\d+) (?P<status>passed|failed|error|skipped)")


def run_tests() -> Dict[str, object]:
    """Execute pytest and capture structured results.

    Returns:
        Mapping with ``passed``, ``failed``, ``total``, ``duration_s``,
        ``collected_at`` and the ``tests`` list.

    Complexity:
        Time dominated by the test suite (a few seconds here); space O(tests).
    """
    root = str(PACKAGE_ROOT)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v", "--no-header", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration = time.perf_counter() - started

    tests: List[Dict[str, object]] = []
    for line in completed.stdout.splitlines():
        match = _RESULT_LINE.match(line.strip())
        if match:
            path = match.group("file").replace("\\", "/")
            tests.append(
                {
                    "file": path,
                    "name": match.group("name"),
                    "status": match.group("status"),
                    "module": os.path.basename(path).replace(".py", ""),
                }
            )

    passed = sum(1 for test in tests if test["status"] == "PASSED")
    failed = sum(1 for test in tests if test["status"] in ("FAILED", "ERROR"))
    skipped = sum(1 for test in tests if test["status"] == "SKIPPED")

    modules: Dict[str, Dict[str, int]] = {}
    for test in tests:
        module = str(test["module"])
        bucket = modules.setdefault(module, {"passed": 0, "failed": 0, "skipped": 0})
        if test["status"] == "PASSED":
            bucket["passed"] += 1
        elif test["status"] in ("FAILED", "ERROR"):
            bucket["failed"] += 1
        else:
            bucket["skipped"] += 1

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": len(tests),
        "duration_s": round(duration, 2),
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_code": completed.returncode,
        "modules": modules,
        "tests": tests,
        "tail": "\n".join(completed.stdout.strip().splitlines()[-3:]),
    }


def main() -> int:
    """CLI entry point.

    Returns:
        ``0`` when the suite passed, ``1`` otherwise.

    Complexity:
        Time dominated by pytest.
    """
    payload = run_tests()
    os.makedirs(str(TABLES_DIR), exist_ok=True)
    path = os.path.join(str(TABLES_DIR), "tests.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"[tests] {payload['passed']} passed, {payload['failed']} failed, "
        f"{payload['skipped']} skipped in {payload['duration_s']}s -> {path}"
    )
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
