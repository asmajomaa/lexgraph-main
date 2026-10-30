"""LEXGRAPH-RAG command line interface.

Usage::

    python main.py --build
    python main.py --query "hockey playoffs goalie" --mode ml --k 10
    python main.py --query "which cases cite 06_1" --mode baseline
    python main.py --interactive
    python main.py --explain "graphics texture" --doc 12
    python main.py --serve            # start the workbench web app

``--build`` parses both corpora and persists a pickled engine under
``cache/engine.pkl`` (see ``engine.py`` for the format rationale).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

from config import DEFAULT_BFS_DEPTH, DEFAULT_PROBE_CLUSTERS, DEFAULT_TOP_K


def _print_result(result) -> None:  # type: ignore[no-untyped-def]
    """Print one agent result as a readable block.

    Args:
        result: An :class:`core.agent.AgentResult`.

    Complexity:
        Time O(k); space O(1).
    """
    print()
    print(f"{'=' * 78}")
    print(f"QUERY   {result.query}")
    print(
        f"PATH    intent={result.intent}  path={result.path}  mode={result.mode}  "
        f"total={result.total_ms:.2f} ms"
    )
    stats = result.stats
    if "candidates_scored" in stats:
        corpus = stats.get("corpus_size", 0.0)
        share = 100.0 * stats["candidates_scored"] / corpus if corpus else 0.0
        print(
            f"STATS   candidates_scored={int(stats['candidates_scored'])} / "
            f"{int(corpus)} ({share:.1f}%)  clusters_probed={int(stats.get('clusters_probed', 0))}"
        )
    if "baseline_overlap" in stats:
        print(
            f"RECALL  {int(stats['baseline_overlap'])}/{int(stats['baseline_k'])} baseline "
            f"results recovered; baseline took {stats.get('baseline_ms', 0.0):.2f} ms"
        )
    print(f"{'-' * 78}")
    if not result.results:
        print("  (no results)")
    for rank, row in enumerate(result.results, start=1):
        score = row.get("score")
        score_text = f"{float(score):7.3f}" if score is not None else "      -"
        depth = row.get("depth")
        depth_text = f" depth={depth}" if depth is not None else ""
        edge = row.get("edge_type")
        edge_text = f" [{edge}]" if edge else ""
        print(f"{rank:>3}. {score_text}{depth_text}{edge_text}  {row.get('external_id')}")
        title = str(row.get("title") or "")
        if title:
            print(f"     {title[:96]}")
        snippet = str(row.get("snippet") or "")
        if snippet:
            print(f"     {snippet[:140]}")
    print(f"{'-' * 78}")
    print(f"ANSWER  {result.answer}")
    print(f"{'=' * 78}")


def cmd_build(args: argparse.Namespace) -> int:
    """Build and cache the engine.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.

    Complexity:
        Time dominated by K-Means; space O(index).
    """
    from engine import build_engine, save_engine

    started = time.perf_counter()
    engine = build_engine(progress=True)
    path = save_engine(engine, args.cache)
    elapsed = time.perf_counter() - started
    print(f"[BUILD] written to {path} in {elapsed:.2f}s")
    print(
        f"[BUILD] news: {engine.news_index.N} docs / {engine.news_index.vocabulary_size()} terms | "
        f"legal: {engine.legal_index.N} cases / {engine.graph.edge_count()} edges"
    )
    print(
        f"[BUILD] k-means: k={engine.kmeans.k} iters={engine.kmeans.n_iter} "
        f"inertia={engine.kmeans.inertia:.2f} purity-vs-labels is reported by benchmarks"
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Run one query through the agent.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.

    Complexity:
        Time dominated by the chosen retrieval path.
    """
    from engine import get_engine

    engine = get_engine(rebuild=False, verbose=True)
    result = engine.query(
        args.query,
        mode=args.mode,
        k=args.k,
        probe_clusters=args.probe_clusters,
        bfs_depth=args.bfs_depth,
        compare_baseline=args.compare,
        edge_type=args.edge_type,
    )
    _print_result(result)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Print the BM25 per-term breakdown for one document.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.

    Complexity:
        Time O(sum of postings lengths for query terms); space O(1).
    """
    from core.bm25 import BM25
    from engine import get_engine

    engine = get_engine(rebuild=False, verbose=False)
    scorer = BM25(engine.news_index)
    print(scorer.explain_text(args.explain, args.doc))
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    """Start the read-eval-print loop.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.

    Commands: ``:mode baseline|ml``, ``:k N``, ``:probe N``, ``:depth N``,
    ``:build``, ``:quit``.

    Complexity:
        Time dominated by the queries the user types.
    """
    from engine import get_engine

    engine = get_engine(rebuild=False, verbose=True)
    mode = args.mode
    k = args.k
    probe = args.probe_clusters
    depth = args.bfs_depth
    edge_filter = args.edge_type
    print("LEXGRAPH-RAG interactive. :help for commands, :quit to exit.")
    while True:
        try:
            line = input("lexgraph> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith(":"):
            command, _, value = line[1:].partition(" ")
            command = command.lower()
            if command in ("quit", "exit", "q"):
                return 0
            if command == "help":
                print(
                    ":mode baseline|ml  :k N  :probe N  :depth N  "
                    ":edges TYPE|any  :build  :quit"
                )
            elif command == "mode" and value in ("baseline", "ml"):
                mode = value
                print(f"mode = {mode}")
            elif command == "k" and value.isdigit():
                k = int(value)
                print(f"k = {k}")
            elif command == "probe" and value.isdigit():
                probe = int(value)
                print(f"probe_clusters = {probe}")
            elif command == "depth" and value.isdigit():
                depth = int(value)
                print(f"bfs_depth = {depth}")
            elif command == "edges":
                edge_filter = None if value in ("", "any") else value
                print(f"edge_type = {edge_filter or 'any'}")
            elif command == "build":
                from engine import build_engine, save_engine

                engine = build_engine(progress=True)
                save_engine(engine)
            else:
                print("unknown command; :help lists them")
            continue
        compare = args.compare or mode == "ml"
        result = engine.query(
            line,
            mode=mode,
            k=k,
            probe_clusters=probe,
            bfs_depth=depth,
            compare_baseline=compare,
            edge_type=edge_filter,
        )
        _print_result(result)


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the workbench web app.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.

    Complexity:
        Time O(1) to bind; requests are handled in the server loop.
    """
    from web.server import serve

    serve(host=args.host, port=args.port, rebuild=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.

    Complexity: O(1).
    """
    parser = argparse.ArgumentParser(
        prog="lexgraph-rag",
        description="LEXGRAPH-RAG: hybrid agentic information retrieval engine",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--build", action="store_true", help="build and cache the indexes")
    mode_group.add_argument("--interactive", action="store_true", help="start the REPL")
    mode_group.add_argument("--serve", action="store_true", help="serve the workbench web app")
    mode_group.add_argument("--explain", metavar="QUERY", help="print a BM25 breakdown")
    parser.add_argument("--query", metavar="TEXT", help="run one query and exit")
    parser.add_argument(
        "--mode", choices=("baseline", "ml"), default="baseline", help="lexical search mode"
    )
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="results to return")
    parser.add_argument(
        "--probe-clusters",
        dest="probe_clusters",
        type=int,
        default=DEFAULT_PROBE_CLUSTERS,
        help="clusters probed on the ML path",
    )
    parser.add_argument(
        "--bfs-depth",
        dest="bfs_depth",
        type=int,
        default=DEFAULT_BFS_DEPTH,
        help="BFS horizon for relational queries",
    )
    parser.add_argument(
        "--edge-type",
        dest="edge_type",
        metavar="CLASS",
        help="restrict the relational BFS to one citation class (e.g. cited, applied)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="also run the baseline and report recall"
    )
    parser.add_argument("--doc", type=int, default=0, help="doc id for --explain")
    parser.add_argument("--cache", metavar="PATH", help="cache file to write on --build")
    parser.add_argument("--host", default="127.0.0.1", help="web host")
    parser.add_argument("--port", type=int, default=8000, help="web port")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.

    Complexity:
        Time dominated by the requested command.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.build:
        return cmd_build(args)
    if args.serve:
        return cmd_serve(args)
    if args.explain:
        return cmd_explain(args)
    if args.interactive:
        return cmd_interactive(args)
    if args.query:
        return cmd_query(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
