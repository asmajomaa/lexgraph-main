"""Generate the architecture and UML diagrams for the report.

Writes three artefacts into ``report/diagrams/``:

* ``architecture.mmd`` / ``architecture.svg`` -- how the three paths and the
  data structures fit together;
* ``uml_class.mmd`` / ``uml_class.svg`` -- the class structure the report's
  section 4 refers to;
* ``index_build.mmd``         -- the build sequence.

Mermaid sources are the primary artefact: they are text, diffable, and render
natively wherever Mermaid is supported.  An SVG fallback is drawn with plain
SVG so the report and the workbench work without a JavaScript runtime.  If a
Mermaid CLI happens to be installed, it is used for the SVG instead of the
hand-drawn fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional, Tuple

from config import REPORT_DIR

DIAGRAM_DIR = os.path.join(str(REPORT_DIR), "diagrams")

ARCHITECTURE_MERMAID = """graph TB
    subgraph Client
        CLI["main.py CLI"]
        WEB["workbench (browser)"]
        API["web/server.py JSON API"]
    end

    subgraph Agent["ReActAgent: Reason -> Act -> Observe"]
        R["REASON<br/>word-level keyword match"]
        A["ACT<br/>dispatch on (intent, mode)"]
        O["OBSERVE / ANSWER<br/>rows + stats + logs"]
    end

    subgraph Paths
        P1["Path 1 RELATIONAL<br/>BFS over citation graph"]
        P2["Path 2 LEXICAL baseline<br/>BM25 over full corpus"]
        P3["Path 3 LEXICAL ml<br/>cluster prune, then BM25"]
    end

    subgraph Structures
        HT["HashSetTable<br/>polynomial hash, chains"]
        PT["PostingsTable<br/>term -> [(doc_id, tf)]"]
        HK["TopKHeap<br/>bounded min-heap"]
        G["CitationGraph<br/>out + in adjacency"]
        SV["SparseVector<br/>sorted parallel arrays"]
    end

    subgraph Core
        IDX["InvertedIndex<br/>doc_lengths, avgdl, N"]
        BM["BM25<br/>k1=1.5, b=0.75"]
        TK["Tokenizer<br/>+ light Porter"]
        VEC["TFIDFVectorizer<br/>top-M by df, L2"]
        KM["KMeans<br/>k-means++, dense centroids"]
        EV["metrics<br/>purity, NMI, ARI"]
    end

    subgraph Data
        NG["20 Newsgroups<br/>18,719 docs"]
        LC["Legal Case Reports<br/>3,890 cases"]
    end

    CLI --> Agent
    WEB --> API
    API --> Agent
    Agent --> R
    R --> A
    A --> P1
    A --> P2
    A --> P3
    P1 --> O
    P2 --> O
    P3 --> O
    O --> CLI
    O --> API

    P1 --> G
    P2 --> BM
    P3 --> KM
    P3 --> BM
    P1 --> IDX
    P2 --> IDX
    BM --> IDX
    IDX --> PT
    IDX --> TK
    PT --> HT
    BM --> HK
    G --> HT
    G --> HK
    KM --> SV
    KM --> VEC
    VEC --> SV
    VEC --> TK
    EV --> KM
    IDX --> NG
    G --> LC
    P1 --> LC
"""

UML_MERMAID = """classDiagram
    class HashSetTable {
        -List _buckets
        -int _capacity
        -int _size
        +int collisions
        +int chain_collisions
        +int probes
        +put(key, value) bool
        +get(key, default) Any
        +contains(key) bool
        +delete(key) bool
        +keys() Iterator
        +load_factor float
    }

    class PostingsTable {
        +add_posting(term, doc_id, freq)
        +get_postings(term) list
        +document_frequency(term) int
    }

    class MinHeap {
        -List _data
        +push(item)
        +pop() RankedItem
        +peek() RankedItem
    }

    class TopKHeap {
        +int k
        +push(score, doc_id) bool
        +get_top_k() list
    }

    class CitationGraph {
        -HashSetTable _out
        -HashSetTable _in
        +add_node(node)
        +add_edge(source, target, type)
        +neighbors(node, direction, edge_type) list
        +bfs_traverse(start, depth, type, direction) list
        +shortest_path(a, b) list
        +connected_component_size(node) int
    }

    class SparseVector {
        +int dimension
        +List indices
        +List values
        +dot(other) float
        +cosine(other) float
        +add(other) SparseVector
        +scale(factor) SparseVector
        +squared_distance(other) float
    }

    class InvertedIndex {
        +PostingsTable postings
        +List doc_lengths
        +Tokenizer tokenizer
        +N int
        +average_document_length float
        +add_document(doc)
        +restrict(candidates) RestrictedView
    }

    class BM25 {
        +float k1
        +float b
        +idf(df, N) float
        +search(query, k) list
        +score_document(query, doc_id) float
        +explain(query, doc_id) list
    }

    class TFIDFVectorizer {
        +int vocab_size
        +Dict vocabulary
        +List idf
        +fit(documents) TFIDFVectorizer
        +transform_text(text) SparseVector
    }

    class KMeans {
        +int k
        +List centroids
        +List labels
        +float inertia
        +fit(vectors) KMeans
        +predict(vector) int
        +predict_top_n(vector, n) list
    }

    class ReActAgent {
        +InvertedIndex news_index
        +CitationGraph graph
        +KMeans kmeans
        +reason(query, mode) tuple
        +act(query, intent, mode, k, probe, depth) tuple
        +observe(path, payload, k) list
        +run(query, mode, k, probe_clusters, bfs_depth) AgentResult
    }

    HashSetTable <|-- PostingsTable
    MinHeap <-- TopKHeap
    InvertedIndex *-- PostingsTable
    InvertedIndex *-- Tokenizer
    CitationGraph *-- HashSetTable
    BM25 --> InvertedIndex
    BM25 ..> TopKHeap
    TFIDFVectorizer ..> SparseVector
    KMeans --> SparseVector
    ReActAgent --> InvertedIndex
    ReActAgent --> CitationGraph
    ReActAgent --> KMeans
    ReActAgent --> TFIDFVectorizer
    ReActAgent --> BM25
"""

BUILD_MERMAID = """sequenceDiagram
    participant C as CLI / API
    participant E as engine.py
    participant L as loaders
    participant I as InvertedIndex
    participant V as TFIDFVectorizer
    participant K as KMeans

    C->>E: build_engine()
    E->>L: load_newsgroups() + load_legal_cases()
    L-->>E: 18,719 docs, 3,890 cases, 3,077 edges
    E->>I: build(news) [tokenise, hash put]
    I-->>E: 112,184 terms
    E->>I: build(legal)
    I-->>E: 94,187 terms
    E->>V: fit(news)
    V-->>E: 5,000-term vocabulary
    E->>V: transform(news)
    V-->>E: 18,719 sparse unit vectors
    E->>K: fit(vectors)
    K-->>E: 20 centroids, labels, inertia
    E-->>C: Engine (pickled to cache/engine.pkl)
"""


def _svg_header(width: int, height: int, title: str) -> List[str]:
    """Open an SVG document with the theme-neutral palette of the report.

    Args:
        width: Canvas width.
        height: Canvas height.
        title: Accessible title.

    Returns:
        SVG lines.

    Complexity: O(1).
    """
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, system-ui, sans-serif">',
        f"<title>{title}</title>",
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#5d6b7a"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    lines: List[str],
    fill: str = "#ffffff",
    stroke: str = "#b9c3cd",
    accent: str = "#1b2430",
) -> List[str]:
    """Draw a labelled box.

    Args:
        x: Left edge.
        y: Top edge.
        w: Width.
        h: Height.
        title: Box heading.
        lines: Detail lines.
        fill: Background.
        stroke: Border colour.
        accent: Heading colour.

    Returns:
        SVG lines.

    Complexity:
        Time O(len(lines)); space O(len(lines)).
    """
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.2"/>',
        f'<text x="{x + 10}" y="{y + 20}" font-size="12.5" font-weight="600" '
        f'fill="{accent}">{title}</text>',
    ]
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 10}" y="{y + 37 + index * 14}" font-size="10.5" '
            f'fill="#5d6b7a">{line}</text>'
        )
    return parts


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str = "", dashed: bool = False) -> List[str]:
    """Draw an arrow with an optional mid-point label.

    Args:
        x1: Start x.
        y1: Start y.
        x2: End x.
        y2: End y.
        label: Optional label.
        dashed: Whether to dash the stroke.

    Returns:
        SVG lines.

    Complexity: O(1).
    """
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    parts = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#5d6b7a" '
        f'stroke-width="1.1"{dash} marker-end="url(#arrow)"/>'
    ]
    if label:
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2 - 5
        parts.append(
            f'<text x="{mid_x}" y="{mid_y}" font-size="9.5" fill="#8a97a5" '
            f'text-anchor="middle">{label}</text>'
        )
    return parts


def _polyline(
    points: List[Tuple[int, int]],
    label: str = "",
    label_at: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Draw an orthogonal polyline with a single arrowhead at its end.

    Used where a straight arrow would cross a box: routing through the empty
    gaps between boxes keeps the diagram readable without moving any box.

    Args:
        points: At least two ``(x, y)`` vertices, in order.
        label: Optional text.
        label_at: Where to place the label; defaults to the midpoint of the
            first segment.

    Returns:
        SVG lines.

    Complexity:
        Time O(len(points)); space O(len(points)).
    """
    if len(points) < 2:
        return []
    parts: List[str] = []
    for index in range(len(points) - 1):
        (x1, y1), (x2, y2) = points[index], points[index + 1]
        last = index == len(points) - 2
        marker = ' marker-end="url(#arrow)"' if last else ""
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#5d6b7a" '
            f'stroke-width="1.1"{marker}/>'
        )
    if label:
        if label_at is None:
            (x1, y1), (x2, y2) = points[0], points[1]
            label_at = ((x1 + x2) // 2, (y1 + y2) // 2 - 5)
        parts.append(
            f'<text x="{label_at[0]}" y="{label_at[1]}" font-size="9.5" '
            f'fill="#8a97a5" text-anchor="middle">{label}</text>'
        )
    return parts


def draw_architecture_svg() -> str:
    """Draw the architecture diagram as standalone SVG.

    Returns:
        The SVG document as a string.

    Complexity:
        Time O(1) -- a fixed number of shapes; space O(1).
    """
    lines = _svg_header(940, 620, "LEXGRAPH-RAG architecture")
    lines.append(
        '<text x="24" y="30" font-size="16" font-weight="700" fill="#1b2430">'
        "LEXGRAPH-RAG: architecture</text>"
    )
    lines.append(
        '<text x="24" y="48" font-size="11" fill="#5d6b7a">'
        "rule-based Reason -> Act -> Observe agent over three retrieval paths</text>"
    )

    lines += _box(24, 70, 250, 96, "Clients", ["CLI  main.py", "Workbench (HTML + JSON API)"], "#f6f8fa")
    lines += _box(24, 190, 250, 120, "ReActAgent", ["reason(): keyword match", "act(): dispatch", "observe(): rows + stats", "run(): result + logs"], "#eef1ff", "#b9b9ee", "#4b4acf")
    lines += _box(24, 334, 250, 76, "Datasets (real)", ["20 Newsgroups: 18,719 docs", "Legal Case Reports: 3,890"], "#f6f8fa")

    lines += _box(340, 70, 250, 110, "Path 2: lexical baseline", ["tokenise query", "BM25 over all postings", "TopKHeap"], "#e6f4f4", "#9fc9c9", "#0f6b6b")
    lines += _box(340, 200, 250, 130, "Path 3: lexical, ML", ["vectorise query", "predict_top_n -> clusters", "ClusterFilter prune", "BM25 over candidates"], "#fbf0dc", "#e0c58a", "#b7791f")
    lines += _box(340, 350, 250, 116, "Path 1: relational", ["resolve case id", "BFS(depth, direction)", "typed edge filter"], "#ecebff", "#b9b9ee", "#4b4acf")

    lines += _box(650, 70, 266, 96, "InvertedIndex", ["HashSetTable postings", "doc_lengths, avgdl, N"], "#ffffff")
    lines += _box(650, 186, 266, 96, "BM25", ["k1=1.5, b=0.75", "TopKHeap (log k)"], "#ffffff")
    lines += _box(650, 302, 266, 82, "KMeans + Vectorizer", ["dense centroids", "L2 sparse vectors"], "#ffffff")
    lines += _box(650, 404, 266, 82, "CitationGraph", ["out + in adjacency", "ring-buffer BFS"], "#ffffff")
    lines += _box(650, 506, 266, 62, "Cache", ["cache/engine.pkl"], "#f6f8fa")

    lines += _arrow(149, 166, 149, 190, "uses")
    # Fan the three path arrows from distinct points on the agent's right edge
    # so the labels do not collide with each other or the Datasets box.
    lines += _arrow(274, 210, 340, 120, "baseline")
    lines += _arrow(274, 248, 340, 262, "ml")
    lines += _arrow(274, 288, 340, 400, "relational")
    # The datasets feed the BUILD (engine.py), not the query path. Route the
    # arrow down the empty gutter at x=300 and across to the Cache row, which
    # is where the built structures land, instead of pointing at a path box.
    lines += _polyline(
        [(274, 360), (300, 360), (300, 537), (650, 537)],
        "loads + builds",
        (312, 430),
    )

    lines += _arrow(590, 125, 650, 118)
    lines += _arrow(590, 265, 650, 234)
    lines += _arrow(590, 340, 650, 343)
    lines += _arrow(590, 408, 650, 445)
    lines += _arrow(783, 166, 783, 186)
    lines.append(
        '<text x="24" y="590" font-size="10" fill="#8a97a5">'
        "Every structure and algorithm is pure Python; numpy/sklearn appear only in benchmarks and tests.</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines)


def draw_uml_svg() -> str:
    """Draw the UML class diagram as standalone SVG.

    Returns:
        The SVG document as a string.

    Complexity:
        Time O(1) -- a fixed number of shapes; space O(1).
    """
    lines = _svg_header(980, 660, "LEXGRAPH-RAG class diagram")
    lines.append(
        '<text x="24" y="30" font-size="16" font-weight="700" fill="#1b2430">'
        "LEXGRAPH-RAG: class structure</text>"
    )

    def class_box(x: int, y: int, w: int, name: str, rows: List[str], colour: str) -> List[str]:
        """Draw one UML class box."""
        height = 30 + len(rows) * 14
        parts = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{height}" rx="4" fill="#ffffff" '
            f'stroke="{colour}" stroke-width="1.3"/>',
            f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="4" fill="{colour}"/>',
            f'<text x="{x + 8}" y="{y + 16}" font-size="11.5" font-weight="700" '
            f'fill="#ffffff">{name}</text>',
        ]
        for index, row in enumerate(rows):
            parts.append(
                f'<text x="{x + 8}" y="{y + 40 + index * 14}" font-size="9.5" '
                f'font-family="JetBrains Mono, Consolas, monospace" fill="#3d4a58">{row}</text>'
            )
        return parts

    lines += class_box(24, 60, 220, "HashSetTable", [
        "put/get/contains/delete",
        "keys, load_factor",
        "collisions, chain_collisions",
        "polynomial hash, chaining",
    ], "#4b4acf")
    lines += class_box(24, 170, 220, "PostingsTable", [
        "add_posting(term,id,tf)",
        "get_postings(term)",
        "document_frequency(term)",
    ], "#4b4acf")
    lines += class_box(24, 280, 220, "MinHeap / TopKHeap", [
        "push / pop / peek",
        "get_top_k()",
        "bounded at size k",
    ], "#4b4acf")
    lines += class_box(24, 400, 220, "CitationGraph", [
        "add_edge(u,v,type)",
        "neighbors(node,dir,type)",
        "bfs_traverse / shortest_path",
    ], "#4b4acf")
    lines += class_box(24, 528, 220, "SparseVector", [
        "indices, values (sorted)",
        "dot / cosine / add / scale",
        "squared_distance",
    ], "#4b4acf")

    lines += class_box(390, 60, 230, "InvertedIndex", [
        "postings: PostingsTable",
        "doc_lengths, N, avgdl",
        "add_document(doc)",
        "restrict(candidates)",
    ], "#0f6b6b")
    lines += class_box(390, 186, 230, "Tokenizer", [
        "tokenize(text)",
        "stop words + Porter-1",
        "min_length",
    ], "#0f6b6b")
    lines += class_box(390, 290, 230, "BM25", [
        "idf(df, N)",
        "search(query, k)",
        "score_document / explain",
    ], "#0f6b6b")
    lines += class_box(390, 410, 230, "TFIDFVectorizer", [
        "fit(documents)",
        "transform_text(text)",
        "vocabulary, idf",
    ], "#0f6b6b")
    lines += class_box(390, 528, 230, "KMeans", [
        "fit / predict / predict_top_n",
        "centroids, labels, inertia",
        "k-means++ seeding",
    ], "#0f6b6b")

    lines += class_box(730, 150, 226, "ReActAgent", [
        "reason(query, mode)",
        "act(...) -> path",
        "observe(path, payload, k)",
        "run(...) -> AgentResult",
    ], "#b7791f")
    lines += class_box(730, 300, 226, "AgentResult", [
        "intent, path, results",
        "logs, timings, stats",
        "to_dict()",
    ], "#b7791f")
    lines += class_box(730, 430, 226, "Engine", [
        "news_index, legal_index",
        "graph, vectorizer, kmeans",
        "query(...) -> AgentResult",
    ], "#b7791f")

    lines += _arrow(244, 200, 390, 110, "composes")
    lines += _arrow(244, 110, 390, 110, "")
    lines += _arrow(244, 320, 390, 350, "used by")
    lines += _arrow(505, 160, 505, 186, "")
    lines += _arrow(505, 240, 505, 290, "")
    lines += _arrow(620, 330, 730, 330, "owns")
    lines += _arrow(620, 440, 730, 470, "")
    lines += _arrow(620, 570, 730, 500, "")
    lines += _arrow(843, 350, 843, 430, "")
    lines.append("</svg>")
    return "\n".join(lines)


def write_diagrams() -> List[str]:
    """Write every Mermaid and SVG diagram.

    Returns:
        Paths written.

    Complexity:
        Time O(1) plus the size of the files; space O(same).
    """
    os.makedirs(DIAGRAM_DIR, exist_ok=True)
    written: List[str] = []
    for name, source in (
        ("architecture.mmd", ARCHITECTURE_MERMAID),
        ("uml_class.mmd", UML_MERMAID),
        ("index_build.mmd", BUILD_MERMAID),
    ):
        path = os.path.join(DIAGRAM_DIR, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        written.append(path)

    for name, drawer in (("architecture.svg", draw_architecture_svg), ("uml_class.svg", draw_uml_svg)):
        path = os.path.join(DIAGRAM_DIR, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(drawer())
        written.append(path)
    return written


def try_mermaid_cli() -> Optional[str]:
    """Render Mermaid with the CLI when it is installed.

    Returns:
        The CLI path, or ``None``.

    Complexity: O(1).
    """
    for candidate in ("mmdc", "mmdc.cmd"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code.

    Complexity:
        Time O(1).
    """
    written = write_diagrams()
    print(f"[diagrams] wrote {len(written)} files into {DIAGRAM_DIR}")
    for path in written:
        print(f"[diagrams]   {os.path.basename(path)}")
    cli = try_mermaid_cli()
    if cli:
        for name in ("architecture", "uml_class"):
            source = os.path.join(DIAGRAM_DIR, f"{name}.mmd")
            target = os.path.join(DIAGRAM_DIR, f"{name}_mermaid.svg")
            try:
                subprocess.run(
                    [cli, "-i", source, "-o", target, "-b", "transparent"],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                print(f"[diagrams] mermaid rendered {target}")
            except (subprocess.SubprocessError, OSError) as error:
                print(f"[diagrams] mermaid CLI failed for {name}: {error}")
    else:
        print("[diagrams] mermaid CLI not found; using the hand-drawn SVG fallbacks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
