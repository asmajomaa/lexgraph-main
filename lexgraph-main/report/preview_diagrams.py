"""Render the hand-drawn SVG diagrams to PNG for a visual check."""

from __future__ import annotations

import os

from config import REPORT_DIR


def main() -> int:
    """Screenshot each SVG diagram at its natural size.

    Returns:
        Exit code.

    Complexity:
        Time O(1).
    """
    from playwright.sync_api import sync_playwright

    directory = os.path.join(str(REPORT_DIR), "diagrams")
    with sync_playwright() as play:
        browser = play.chromium.launch()
        for name, width, height in (("architecture", 940, 620), ("uml_class", 980, 660)):
            page = browser.new_page(viewport={"width": width, "height": height})
            url = "file:///" + os.path.join(directory, name + ".svg").replace("\\", "/")
            page.goto(url)
            page.wait_for_timeout(300)
            page.screenshot(path=os.path.join(directory, name + "_preview.png"))
            page.close()
            print(f"[preview] {name}_preview.png")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
