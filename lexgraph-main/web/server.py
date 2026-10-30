"""HTTP JSON API backing the LEXGRAPH-RAG workbench.

Standard library only (``http.server``): no Flask, no external dependency, so
``python main.py --serve`` works on a bare Python install.

Endpoints
---------
``GET  /``               the workbench page (``web/workbench.html``)
``GET  /api/info``       corpus, index and cluster metadata
``POST /api/query``      run one query through the agent
``GET  /api/complexity`` the measured complexity table (from the report data)
``GET  /api/tests``      the last pytest run's structured results
``GET  /api/figures``    the report's figure and table inventory
``POST /api/build``      rebuild the engine (guarded: long-running)
``GET  /static/<file>``  report figures served for the Complexity/Results tabs

Every handler returns JSON with an ``error`` key on failure and an appropriate
status code, so the frontend can render failures instead of failing silently.

Security note: the server binds to ``127.0.0.1`` by default and rejects
requests whose ``Host`` header is not loopback, which blocks DNS-rebinding
attacks against a developer's browser.  It is a local development tool, not a
production service, and says so in the UI.
"""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

from config import PACKAGE_ROOT, REPORT_DIR, TABLES_DIR

#: Directory holding the served static page and its assets.
WEB_DIR: str = os.path.join(str(PACKAGE_ROOT), "web")

#: Directories that may be served read-only as ``/static``.
STATIC_ROOTS: Dict[str, str] = {
    "figures": os.path.join(str(REPORT_DIR), "figures"),
    "tables": os.path.join(str(REPORT_DIR), "tables"),
    "diagrams": os.path.join(str(REPORT_DIR), "diagrams"),
}

#: Allowed static file extensions.
STATIC_EXTENSIONS = {".png", ".svg", ".csv", ".md", ".json", ".css", ".js"}

#: A single engine instance shared by all requests; the agent is read-only at
#: query time so concurrent requests are safe without a lock.
_ENGINE: Optional[object] = None
_ENGINE_LOCK = threading.Lock()
_BUILD_STATE: Dict[str, object] = {"running": False, "started": 0.0, "finished": 0.0, "error": ""}


def get_engine() -> object:
    """Return the process-wide engine, loading it on first use.

    Returns:
        The :class:`engine.Engine`.

    Complexity:
        Time O(1) after the first call (which unpickles the cache);
        space O(engine).
    """
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                from engine import get_engine as load

                _ENGINE = load(verbose=False)
    return _ENGINE


def _quartiles(values: List[float]) -> Dict[str, float]:
    """Five-number summary plus mean, for latency panels.

    Args:
        values: Measurements (milliseconds).

    Returns:
        Mapping with ``min``, ``p50``, ``p90``, ``p99``, ``max``, ``mean``.

    Complexity:
        Time O(n log n); space O(n).
    """
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(values)
    count = len(ordered)

    def at(fraction: float) -> float:
        """Nearest-rank percentile."""
        index = min(count - 1, max(0, int(round(fraction * (count - 1)))))
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / count,
    }


def info_payload() -> Dict[str, object]:
    """Metadata for the workbench header and the Corpora panel.

    Returns:
        JSON-ready mapping.

    Complexity:
        Time O(T) for collection statistics; space O(1).
    """
    engine = get_engine()
    news_index = engine.news_index  # type: ignore[attr-defined]
    legal_index = engine.legal_index  # type: ignore[attr-defined]
    graph = engine.graph  # type: ignore[attr-defined]
    kmeans = engine.kmeans  # type: ignore[attr-defined]
    news_stats = news_index.collection_statistics()
    legal_stats = legal_index.collection_statistics()
    return {
        "demo": False,
        "news": {
            "documents": news_index.N,
            "categories": len(engine.news.categories),  # type: ignore[attr-defined]
            "source": engine.news.source,  # type: ignore[attr-defined]
            "vocabulary": int(news_stats["vocabulary"]),
            "tokens": int(news_stats["tokens"]),
            "avgdl": round(news_stats["avgdl"], 2),
            "postings": int(news_stats["postings"]),
            "load_factor": round(news_stats["load_factor"], 3),
            "collisions": int(news_stats["collisions"]),
        },
        "legal": {
            "cases": legal_index.N,
            "source": engine.legal.source,  # type: ignore[attr-defined]
            "edges": graph.edge_count(),
            "edge_classes": graph.edge_type_counts(),
            "vocabulary": int(legal_stats["vocabulary"]),
        },
        "kmeans": {
            "k": kmeans.k,
            "iterations": kmeans.n_iter,
            "converged": kmeans.converged,
            "inertia": round(kmeans.inertia, 2),
            "vocabulary": engine.vectorizer.dimension,  # type: ignore[attr-defined]
            "seed": kmeans.seed,
        },
        "build": {
            "running": bool(_BUILD_STATE["running"]),
            "error": _BUILD_STATE["error"],
        },
    }


def query_payload(body: Dict[str, object]) -> Dict[str, object]:
    """Run one query.

    Args:
        body: Request body with ``query``, ``mode``, ``k``,
            ``probe_clusters`` and an optional ``edge_type`` citation-class
            filter for relational queries.

    Returns:
        The agent result as JSON.

    Raises:
        ValueError: On a missing or empty query.

    Complexity:
        Time dominated by the chosen retrieval path; space O(corpus) transient.
    """
    query = str(body.get("query", "")).strip()
    if not query:
        raise ValueError("query must not be empty")
    mode = str(body.get("mode", "baseline"))
    if mode not in ("baseline", "ml"):
        raise ValueError("mode must be 'baseline' or 'ml'")
    try:
        k = int(body.get("k", 10))
    except (TypeError, ValueError):
        k = 10
    try:
        probe = int(body.get("probe_clusters", 2))
    except (TypeError, ValueError):
        probe = 2
    k = max(1, min(k, 100))
    probe = max(1, min(probe, 20))
    edge_type = body.get("edge_type")
    edge_filter = str(edge_type).strip() if edge_type else None
    if edge_filter in ("", "any", "None"):
        edge_filter = None

    started = time.perf_counter()
    engine = get_engine()
    result = engine.query(  # type: ignore[attr-defined]
        query,
        mode=mode,
        k=k,
        probe_clusters=probe,
        compare_baseline=True,
        edge_type=edge_filter,
    )
    payload = result.to_dict()
    payload["server_ms"] = (time.perf_counter() - started) * 1000.0
    return payload


def complexity_payload() -> Dict[str, object]:
    """Measured complexity data for the Complexity panel.

    Returns:
        Mapping with the operation list plus the measured scaling series from
        ``report/tables/summary.json`` when a benchmark run has produced it.

    Complexity:
        Time O(file size); space O(file size).
    """
    summary: Dict[str, object] = {}
    path = os.path.join(str(TABLES_DIR), "summary.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            summary = {}

    engine = get_engine()
    operations = [
        {
            "name": "HashTable.put",
            "structure": "hash table",
            "time": "O(1) expected, O(n) worst",
            "space": "O(1) amortised",
            "why": "Polynomial hash of the key, then one chain walk; resize doubles "
            "capacity and costs O(n) amortised over n inserts.",
        },
        {
            "name": "HashTable.get",
            "structure": "hash table",
            "time": "O(1) expected",
            "space": "O(1)",
            "why": "Same bucket probe; average chain length stays near 1.5 because "
            "the load factor is capped at 0.75.",
        },
        {
            "name": "TopKHeap.push",
            "structure": "bounded min-heap",
            "time": "O(log k)",
            "space": "O(1)",
            "why": "A candidate only enters when it beats the root; sift-up costs "
            "one root-to-leaf path.",
        },
        {
            "name": "BM25.search",
            "structure": "inverted index + heap",
            "time": "O(P log k)",
            "space": "O(N + k)",
            "why": "P postings are visited once and accumulated into a dense score "
            "array, then the heap keeps the best k.",
        },
        {
            "name": "BFS",
            "structure": "graph + ring queue",
            "time": "O(V + E)",
            "space": "O(V)",
            "why": "Every node is enqueued once and every adjacency entry scanned "
            "once; the visited bitmap makes the check O(1).",
        },
        {
            "name": "KMeans.fit",
            "structure": "sparse vectors + dense centroids",
            "time": "O(n k d T)",
            "space": "O(k d + n)",
            "why": "n documents x k centroids x d dimensions per assignment, "
            "repeated T iterations. The sparse-dot identity cuts d to nnz per "
            "document.",
        },
        {
            "name": "KMeans.predict_top_n",
            "structure": "dense centroids",
            "time": "O(k nnz + k log k)",
            "space": "O(k)",
            "why": "One sparse-dense dot per centroid, then a sort of k distances.",
        },
        {
            "name": "agent cluster pruning",
            "structure": "cluster filter",
            "time": "O(k + P)",
            "space": "O(k)",
            "why": "The per-query mask is one byte per cluster, and membership is two "
            "array indices rather than a hash.",
        },
    ]
    news = engine.news_index  # type: ignore[attr-defined]
    return {
        "operations": operations,
        "corpus": {
            "documents": news.N,
            "vocabulary": news.vocabulary_size(),
            "avgdl": round(news.average_document_length, 2),
            "total_tokens": news.total_length,
        },
        "measured": {
            "scalability": summary.get("scalability", {}),
            "graph": summary.get("graph", {}),
            "baseline_vs_ml": summary.get("query_time", {}),
            "kmeans": summary.get("kmeans", {}),
            "compare": summary.get("compare", {}),
        },
    }


def tests_payload() -> Dict[str, object]:
    """Structured results of the most recent test run.

    The payload comes from ``report/tables/tests.json``, written by
    ``web/run_tests.py``.  When it is missing the panel explains how to
    produce it rather than inventing results.

    Returns:
        Mapping with ``available``, ``passed``, ``failed``, ``duration`` and
        the per-test list.

    Complexity:
        Time O(file size); space O(file size).
    """
    path = os.path.join(str(TABLES_DIR), "tests.json")
    if not os.path.exists(path):
        return {
            "available": False,
            "hint": "Run `python -m web.run_tests` to generate report/tables/tests.json.",
            "tests": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "hint": f"could not read tests.json: {error}", "tests": []}
    payload["available"] = True
    return payload


def figures_payload() -> Dict[str, object]:
    """Inventory of report figures and tables, for the Results panel.

    Returns:
        Mapping with ``figures`` and ``tables`` lists of ``{name, url, bytes}``.

    Complexity:
        Time O(files); space O(files).
    """
    def scan(folder: str, prefix: str, extensions: Tuple[str, ...]) -> List[Dict[str, object]]:
        """List matching files in ``folder`` with their static URL."""
        directory = STATIC_ROOTS[folder]
        if not os.path.isdir(directory):
            return []
        found: List[Dict[str, object]] = []
        for name in sorted(os.listdir(directory)):
            if not name.lower().endswith(extensions):
                continue
            full = os.path.join(directory, name)
            found.append(
                {
                    "name": name,
                    "url": f"/static/{prefix}/{urllib.parse.quote(name)}",
                    "bytes": os.path.getsize(full),
                }
            )
        return found

    return {
        "figures": scan("figures", "figures", (".png", ".svg")),
        "tables": scan("tables", "tables", (".md", ".csv")),
        "diagrams": scan("diagrams", "diagrams", (".svg", ".png", ".md")),
    }


def rebuild_payload() -> Dict[str, object]:
    """Kick off an engine rebuild in a background thread.

    Returns:
        ``{"started": True}`` or a message when one is already running.

    Complexity:
        Time O(1) to start; the rebuild itself runs on a worker thread.
    """
    if _BUILD_STATE["running"]:
        return {"started": False, "reason": "a build is already running"}
    with _ENGINE_LOCK:
        _BUILD_STATE["running"] = True
        _BUILD_STATE["started"] = time.time()
        _BUILD_STATE["error"] = ""

    def worker() -> None:
        """Rebuild the engine and swap it in."""
        global _ENGINE
        try:
            from engine import build_engine, save_engine

            rebuilt = build_engine(verbose=False, progress=False)
            save_engine(rebuilt)
            _ENGINE = rebuilt
        except Exception as error:  # noqa: BLE001 - surface any failure to the UI
            _BUILD_STATE["error"] = f"{type(error).__name__}: {error}"
        finally:
            _BUILD_STATE["running"] = False
            _BUILD_STATE["finished"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}


class WorkbenchHandler(BaseHTTPRequestHandler):
    """Request router for the workbench."""

    server_version = "LexgraphRAG/1.0"

    # -- helpers -----------------------------------------------------------

    def _host_is_local(self) -> bool:
        """Reject non-loopback ``Host`` headers (DNS-rebinding defence).

        Returns:
            ``True`` when the request may be served.

        Complexity:
            Time O(len(host)); space O(1).
        """
        host = (self.headers.get("Host") or "").split(":")[0].strip().lower()
        return host in ("127.0.0.1", "localhost", "::1", "")

    def _send_json(self, payload: object, status: int = 200) -> None:
        """Write a JSON response.

        Args:
            payload: JSON-serialisable object.
            status: HTTP status code.

        Complexity:
            Time O(response size); space O(response size).
        """
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: Optional[str] = None) -> None:
        """Stream a file to the client.

        Args:
            path: Absolute file path under an allowed root.
            content_type: Override for the detected MIME type.

        Complexity:
            Time O(file size); space O(chunk size).
        """
        try:
            with open(path, "rb") as handle:
                body = handle.read()
        except OSError:
            self._send_json({"error": "file not found"}, 404)
            return
        mime = content_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
        textual = mime.startswith("text/") or mime in (
            "application/json",
            "image/svg+xml",
        )
        if textual and "charset" not in mime:
            mime += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static_path(self, url_path: str) -> Optional[str]:
        """Resolve a ``/static/...`` URL to a safe absolute path.

        Args:
            url_path: Request path beginning with ``/static/``.

        Returns:
            An absolute path inside an allowed root, or ``None``.

        Complexity:
            Time O(len(path)); space O(1).
        """
        relative = urllib.parse.unquote(url_path[len("/static/") :])
        parts = [part for part in relative.replace("\\", "/").split("/") if part]
        if len(parts) < 2 or parts[0] not in STATIC_ROOTS:
            return None
        folder = STATIC_ROOTS[parts[0]]
        name = parts[-1]
        if name != os.path.basename(name) or ".." in parts:
            return None
        if os.path.splitext(name)[1].lower() not in STATIC_EXTENSIONS:
            return None
        candidate = os.path.join(folder, name)
        return candidate if os.path.isfile(candidate) else None

    def log_message(self, format_string: str, *args: object) -> None:
        """Quieter access log: method, path and status only.

        Args:
            format_string: Unused ``http.server`` format string.
            *args: Unused format arguments.

        Complexity: O(1).
        """
        return

    # -- verbs -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Serve the page, the JSON API and static report assets.

        Complexity:
            Time dominated by the requested endpoint; space O(response size).
        """
        if not self._host_is_local():
            self._send_json({"error": "non-loopback Host header rejected"}, 403)
            return
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html", "/workbench.html"):
            page = os.path.join(WEB_DIR, "workbench.html")
            self._send_file(page, "text/html; charset=utf-8")
            return
        if route.startswith("/static/"):
            resolved = self._static_path(route)
            if resolved is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_file(resolved)
            return
        try:
            if route == "/api/info":
                self._send_json(info_payload())
            elif route == "/api/complexity":
                self._send_json(complexity_payload())
            elif route == "/api/tests":
                self._send_json(tests_payload())
            elif route == "/api/figures":
                self._send_json(figures_payload())
            elif route == "/api/health":
                self._send_json({"ok": True, "time": time.time()})
            else:
                self._send_json({"error": f"unknown route {route}"}, 404)
        except Exception as error:  # noqa: BLE001 - never drop the connection
            self._send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Handle ``/api/query`` and ``/api/build``.

        Complexity:
            Time dominated by the requested operation; space O(request body).
        """
        if not self._host_is_local():
            self._send_json({"error": "non-loopback Host header rejected"}, 403)
            return
        route = urllib.parse.urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "request body must be JSON"}, 400)
            return
        if not isinstance(body, dict):
            self._send_json({"error": "request body must be a JSON object"}, 400)
            return

        try:
            if route == "/api/query":
                self._send_json(query_payload(body))
            elif route == "/api/build":
                self._send_json(rebuild_payload())
            else:
                self._send_json({"error": f"unknown route {route}"}, 404)
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)
        except Exception as error:  # noqa: BLE001 - never drop the connection
            self._send_json({"error": f"{type(error).__name__}: {error}"}, 500)


def serve(host: str = "127.0.0.1", port: int = 8000, rebuild: bool = False) -> None:
    """Start the workbench server.

    Args:
        host: Bind address.  Defaults to loopback; the handler rejects
            non-loopback ``Host`` headers regardless.
        port: Bind port.
        rebuild: Rebuild the engine before serving.

    Complexity:
        Time O(engine build) when ``rebuild`` is set, then O(1) to bind.
    """
    if rebuild:
        from engine import build_engine, save_engine

        print("[serve] rebuilding engine...")
        save_engine(build_engine(verbose=True, progress=True))

    print("[serve] loading engine...")
    get_engine()
    server = ThreadingHTTPServer((host, port), WorkbenchHandler)
    print(f"[serve] LEXGRAPH-RAG workbench -> http://{host}:{port}/")
    print("[serve] endpoints: /api/info /api/query /api/tests /api/complexity /api/figures")
    print("[serve] Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down")
    finally:
        server.server_close()


__all__ = ["serve", "WorkbenchHandler", "info_payload", "query_payload"]
