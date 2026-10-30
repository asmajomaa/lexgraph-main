"""Developer utility: screenshot the workbench in both themes and at phone width.

Not part of the deliverable's test surface; it exists so the UI can be
verified by looking at it rather than by assertion.

Usage::

    python -m web.screenshot            # writes report/diagrams/ui_*.png
    python -m web.screenshot --port 9000
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import List, Optional

from config import REPORT_DIR


def capture(port: int = 8766, wait: float = 16.0) -> List[str]:
    """Start the server, screenshot each view, and return the written paths.

    Args:
        port: Port to bind.
        wait: Seconds to wait for the engine to load before the first page.

    Returns:
        The list of files written.

    Complexity:
        Time dominated by engine loading (one pickle read) plus page rendering.
    """
    from playwright.sync_api import sync_playwright

    from web.server import serve

    threading.Thread(
        target=serve, kwargs={"host": "127.0.0.1", "port": port}, daemon=True
    ).start()
    time.sleep(wait)

    out_dir = os.path.join(str(REPORT_DIR), "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    errors: List[str] = []
    base = f"http://127.0.0.1:{port}"

    with sync_playwright() as play:
        browser = play.chromium.launch()

        for scheme in ("light", "dark"):
            page = browser.new_page(viewport={"width": 1280, "height": 1500}, color_scheme=scheme)
            page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
            page.goto(f"{base}/#search", wait_until="networkidle")
            page.fill("#q", "which cases cite 06_584")
            page.click("#go")
            page.wait_for_selector("ol.res li", timeout=30000)
            page.wait_for_timeout(700)
            path = os.path.join(out_dir, f"ui_search_{scheme}.png")
            page.screenshot(path=path, full_page=True)
            written.append(path)
            page.close()

        for view in ("tests", "complexity", "results"):
            page = browser.new_page(viewport={"width": 1280, "height": 1400})
            page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
            page.goto(f"{base}/#{view}", wait_until="networkidle")
            page.wait_for_timeout(2500)
            path = os.path.join(out_dir, f"ui_{view}.png")
            page.screenshot(path=path, full_page=True)
            written.append(path)
            page.close()

        page = browser.new_page(viewport={"width": 400, "height": 900})
        page.goto(f"{base}/#search", wait_until="networkidle")
        page.wait_for_timeout(1200)
        path = os.path.join(out_dir, "ui_phone.png")
        page.screenshot(path=path, full_page=True)
        written.append(path)
        overflow = page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        print(f"[screenshot] phone horizontal overflow: {overflow}")
        page.close()
        browser.close()

    print(f"[screenshot] page errors: {errors if errors else 'none'}")
    for path in written:
        print(f"[screenshot] wrote {path}")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Screenshot the workbench")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--wait", type=float, default=16.0)
    args = parser.parse_args(argv)
    capture(args.port, args.wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
