"""ReAct-style agent: Reason -> Act -> Observe -> Answer.

The agent owns no index of its own; it holds references to the two indexes
(newsgroups, legal), the citation graph, the K-Means model and the
document-to-cluster map, and routes each query to one of three paths:

===================  ==========================================================
Intent / mode        Path
===================  ==========================================================
RELATIONAL           Extract a case id, then BFS over the citation graph
LEXICAL / baseline   BM25 over the full newsgroups corpus
LEXICAL / ml         K-Means prunes to the nearest ``probe_clusters`` clusters,
                     then BM25 over those candidates only
===================  ==========================================================

Every step emits a structured log line (``[AGENT][REASON] ...``) so a console
transcript can be pasted straight into the report.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from config import (
    DEFAULT_BFS_DEPTH,
    DEFAULT_PROBE_CLUSTERS,
    RELATIONAL_KEYWORDS,
)
from core.bm25 import BM25
from core.inverted_index import ClusterFilter, InvertedIndex
from data.load_legal_cases import LegalCorpus
from ml.kmeans import KMeans
from ml.vectorizer import TFIDFVectorizer
from structures.graph import CitationGraph
from structures.hash_table import HashSetTable


class AgentResult:
    """Structured outcome of one agent run.

    Attributes:
        query: The original query.
        intent: ``"RELATIONAL"`` or ``"LEXICAL"``.
        path: ``"relational"``, ``"baseline"`` or ``"ml"``.
        mode: Requested mode (``"baseline"``/``"ml"``).
        results: Ranked result dicts ready for the API/UI.
        logs: Structured log lines.
        timings: Stage name -> milliseconds.
        stats: ``corpus_size``, ``candidates_scored``, ``postings_visited``,
            ``clusters_probed``, ``graph_nodes``, ``graph_edges``.
        answer: One-paragraph natural-language answer.
    """

    __slots__ = (
        "query",
        "intent",
        "path",
        "mode",
        "results",
        "logs",
        "timings",
        "stats",
        "answer",
    )

    def __init__(
        self,
        query: str,
        intent: str,
        path: str,
        mode: str,
        results: List[Dict[str, object]],
        logs: List[str],
        timings: Dict[str, float],
        stats: Dict[str, float],
        answer: str,
    ) -> None:
        """Create one result record.

        Args:
            query: The original query.
            intent: ``"RELATIONAL"`` or ``"LEXICAL"``.
            path: ``"relational"``, ``"baseline"`` or ``"ml"``.
            mode: Requested mode.
            results: Ranked result dicts.
            logs: Structured log lines.
            timings: Stage name -> milliseconds.
            stats: Measured counters.
            answer: Natural-language answer.

        Complexity: Time O(1); space O(1) beyond the referenced sequences.
        """
        self.query = query
        self.intent = intent
        self.path = path
        self.mode = mode
        self.results = results
        self.logs = logs
        self.timings = timings
        self.stats = stats
        self.answer = answer

    @property
    def total_ms(self) -> float:
        """Total wall time for the run in milliseconds.

        Complexity: O(1).
        """
        return self.timings.get("total", 0.0)

    def to_dict(self) -> Dict[str, object]:
        """JSON-ready projection matching the web API contract.

        Returns:
            Mapping with ``intent``, ``path``, ``mode``, ``results``,
            ``logs``, ``timings``, ``stats``, ``total_ms`` and ``answer``.

        Complexity:
            Time O(number of results); space O(same).
        """
        return {
            "query": self.query,
            "intent": self.intent,
            "path": self.path,
            "mode": self.mode,
            "results": self.results,
            "logs": self.logs,
            "timings": self.timings,
            "total_ms": self.total_ms,
            "stats": self.stats,
            "answer": self.answer,
        }

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return (
            f"AgentResult({self.intent}/{self.path}, "
            f"results={len(self.results)}, {self.total_ms:.2f} ms)"
        )


def is_relational(query: str) -> bool:
    """Rule-based intent classifier.

    Args:
        query: Raw query.

    Returns:
        ``True`` when a relational keyword appears as a word in the query.

    Complexity:
        Time O(len(query)); space O(1).

    The match is word-based, not substring: ``"precedence"`` must not fire the
    ``"precedent"`` rule, and ``"excited"`` must not fire ``"cite"``.
    """
    lowered = query.lower()
    words: HashSetTable = HashSetTable()
    current: List[str] = []
    for char in lowered:
        if "a" <= char <= "z":
            current.append(char)
        else:
            if current:
                words.put("".join(current), 1)
                current = []
    if current:
        words.put("".join(current), 1)
    for keyword in RELATIONAL_KEYWORDS:
        if words.contains(keyword):
            return True
    return False


class ReActAgent:
    """The Reason -> Act -> Observe agent.

    Attributes:
        news_index: Inverted index over 20 Newsgroups.
        legal_index: Inverted index over the legal cases.
        graph: Citation graph over the legal cases.
        legal_corpus: Case metadata and the alias resolver.
        kmeans: Fitted cluster model over the newsgroups vectors.
        vectorizer: Fitted TF-IDF vectorizer shared with ``kmeans``.
        clusters: ``doc_id -> cluster`` for the newsgroups corpus.
        cluster_members: ``cluster -> [doc_id, ...]``.

    Space: O(index size) for the referenced structures.
    """

    __slots__ = (
        "news_index",
        "legal_index",
        "graph",
        "legal_corpus",
        "kmeans",
        "vectorizer",
        "clusters",
        "cluster_members",
        "_external_index",
        "news_scorer",
        "legal_scorer",
        "verbose",
    )

    def __init__(
        self,
        news_index: InvertedIndex,
        legal_index: InvertedIndex,
        graph: CitationGraph,
        legal_corpus: Optional[LegalCorpus] = None,
        kmeans: Optional[KMeans] = None,
        vectorizer: Optional[TFIDFVectorizer] = None,
        clusters: Optional[Sequence[int]] = None,
        verbose: bool = True,
    ) -> None:
        """Wire the agent to its data structures.

        Args:
            news_index: Newsgroups inverted index.
            legal_index: Legal inverted index.
            graph: Citation graph.
            legal_corpus: Legal metadata (needed to resolve case ids).
            kmeans: Fitted clusterer for the ML path.
            vectorizer: Fitted vectorizer matching ``kmeans``.
            clusters: Cluster assignment per newsgroups document.
            verbose: Whether :meth:`run` prints the logs to stdout.

        Complexity: O(n) to build the inverse cluster map.
        """
        self.news_index = news_index
        self.legal_index = legal_index
        self.graph = graph
        self.legal_corpus = legal_corpus
        self.kmeans = kmeans
        self.vectorizer = vectorizer
        self.clusters = list(clusters) if clusters is not None else []
        self.news_scorer = BM25(news_index)
        self.legal_scorer = BM25(legal_index)
        self.verbose = verbose
        members: HashSetTable = HashSetTable()
        if kmeans is not None:
            for doc_id, cluster in enumerate(self.clusters):
                bucket = members.get(cluster)
                if bucket is None:
                    members.put(cluster, [doc_id])
                else:
                    bucket.append(doc_id)
        self.cluster_members = members
        # Built lazily: the relational path needs it, the lexical paths do not,
        # and it is 3,890 hash inserts (13.95 ms, measured) that would otherwise
        # be paid on every relational query.
        self._external_index: Optional[HashSetTable] = None

    # -- Reason ------------------------------------------------------------

    def reason(self, query: str, mode: str) -> Tuple[str, List[str]]:
        """Classify the query intent.

        Args:
            query: Raw query.
            mode: Requested mode (recorded in the log for context).

        Returns:
            ``(intent, log_lines)``.

        Complexity:
            Time O(len(query)); space O(1).
        """
        relational = is_relational(query)
        intent = "RELATIONAL" if relational else "LEXICAL"
        matched = [
            keyword
            for keyword in RELATIONAL_KEYWORDS
            if keyword in query.lower().split()
        ] if relational else []
        if relational:
            detail = f"relational keyword(s) {matched} -> citation-graph path"
        else:
            detail = f"no relational keyword (mode={mode}) -> lexical path"
        lines = [f"[AGENT][REASON] intent={intent}: {detail}"]
        return intent, lines

    # -- Act ---------------------------------------------------------------

    def act(
        self,
        query: str,
        intent: str,
        mode: str,
        k: int,
        probe_clusters: int,
        bfs_depth: int,
        edge_type: Optional[str] = None,
    ) -> Tuple[str, Dict[str, object]]:
        """Dispatch to the path chosen by :meth:`reason`.

        Args:
            query: Raw query.
            intent: Result of :meth:`reason`.
            mode: ``"baseline"`` or ``"ml"``.
            k: Number of results.
            probe_clusters: Clusters to probe on the ML path.
            bfs_depth: BFS horizon on the relational path.
            edge_type: Optional citation-class filter for the relational BFS
                (``"cited"``, ``"applied"``, ...).  ``None`` follows every class.

        Returns:
            ``(path, payload)`` where the payload carries ranked ids, stats and
            log lines.

        Complexity:
            Time O(P log K) on the two lexical paths and O(V + E) on the
            relational path, where P is the number of postings scored, V/E the
            graph size and K the requested result count; space O(D + K) for the
            dense score array (D = corpus size) plus the result heap.
        """
        if intent == "RELATIONAL":
            return "relational", self._act_relational(query, k, bfs_depth, edge_type)
        if mode == "ml" and self.kmeans is not None and self.vectorizer is not None:
            return "ml", self._act_ml(query, k, probe_clusters)
        return "baseline", self._act_baseline(query, k)

    def _act_baseline(self, query: str, k: int) -> Dict[str, object]:
        """BM25 over the full newsgroups corpus.

        Args:
            query: Raw query.
            k: Number of results.

        Returns:
            Payload with ``results``, ``stats`` and ``logs``.

        Complexity:
            Time O(P log K); space O(D + K).
        """
        started = time.perf_counter()
        ranked = self.news_scorer.search(query, k)
        elapsed = (time.perf_counter() - started) * 1000.0
        stats = {
            "corpus_size": float(self.news_index.N),
            "candidates_scored": float(self.news_scorer.scored_documents),
            "postings_visited": float(self.news_scorer.last_stats.get("postings_visited", 0.0)),
            "terms": float(self.news_scorer.last_stats.get("terms", 0.0)),
            "clusters_probed": 0.0,
        }
        logs = [
            f"[AGENT][ACT] lexical/baseline: BM25 over all {self.news_index.N} documents",
            f"[AGENT][OBSERVE] scored {self.news_scorer.scored_documents} documents "
            f"({self._pct(stats['candidates_scored'], stats['corpus_size'])}) in {elapsed:.2f} ms",
        ]
        return {"ranked": ranked, "stats": stats, "logs": logs, "elapsed_ms": elapsed}

    def _act_ml(self, query: str, k: int, probe_clusters: int) -> Dict[str, object]:
        """K-Means pruning followed by BM25 over the surviving candidates.

        Args:
            query: Raw query.
            k: Number of results.
            probe_clusters: Clusters to probe (>= 1).

        Returns:
            Payload with ``results``, ``stats`` and ``logs``.

        Complexity:
            Time O(k_clusters * nnz_q) to assign + O(P_candidate log K) to score
            + O(n_candidates) to build the candidate set; space O(n_candidates).
        """
        started = time.perf_counter()
        assert self.vectorizer is not None and self.kmeans is not None
        query_vector = self.vectorizer.transform_text(query)
        vectorize_ms = (time.perf_counter() - started) * 1000.0

        probe_started = time.perf_counter()
        nearest = self.kmeans.predict_top_n(query_vector, max(1, probe_clusters))
        clusters = [cluster for cluster, _distance in nearest]
        candidates = ClusterFilter(self.clusters, clusters, n_clusters=self.kmeans.k)
        probe_ms = (time.perf_counter() - probe_started) * 1000.0

        view = self.news_index.restrict(candidates)
        scorer = BM25(view)
        score_started = time.perf_counter()
        ranked = scorer.search(query, k)
        score_ms = (time.perf_counter() - score_started) * 1000.0
        elapsed = (time.perf_counter() - started) * 1000.0

        clusters_text = ", ".join(str(cluster) for cluster in clusters)
        pool_size = sum(len(self.cluster_members.get(cluster, [])) for cluster in clusters)
        stats = {
            "corpus_size": float(self.news_index.N),
            "candidates_scored": float(scorer.scored_documents),
            "candidate_pool": float(pool_size),
            "postings_visited": float(scorer.last_stats.get("postings_visited", 0.0)),
            "terms": float(scorer.last_stats.get("terms", 0.0)),
            "clusters_probed": float(len(nearest)),
            "vectorize_ms": vectorize_ms,
            "probe_ms": probe_ms,
            "score_ms": score_ms,
        }
        logs = [
            f"[AGENT][ACT] lexical/ml: vectorised query (nnz={query_vector.nnz}), "
            f"probing {len(nearest)} cluster(s) [{clusters_text}] in {probe_ms:.2f} ms",
            f"[AGENT][ACT] candidate pool = {pool_size} of {self.news_index.N} "
            f"documents ({self._pct(float(pool_size), float(self.news_index.N))}); "
            f"BM25 over candidates in {score_ms:.2f} ms",
            f"[AGENT][OBSERVE] scored {scorer.scored_documents} documents in {elapsed:.2f} ms",
        ]
        return {"ranked": ranked, "stats": stats, "logs": logs, "elapsed_ms": elapsed}

    def _act_relational(
        self,
        query: str,
        k: int,
        bfs_depth: int,
        edge_type: Optional[str] = None,
    ) -> Dict[str, object]:
        """BFS over the citation graph, seeded from a case id in the query.

        Args:
            query: Raw query.
            k: Maximum number of connected cases to return.
            bfs_depth: BFS horizon in hops.
            edge_type: Optional citation-class filter.

        Returns:
            Payload with ``results``, ``stats`` and ``logs``.

        Complexity:
            Time O(len(query)) to resolve the seed + O(V + E) for the bounded
            BFS + O(P log 1) for the fallback lexical lookup; space O(V).
        """
        started = time.perf_counter()
        logs: List[str] = []
        seed_id: Optional[int] = None
        resolution = "query"

        if self.legal_corpus is not None:
            seed_id = self._extract_case_id(query)
            logs.append(
                f"[AGENT][ACT] relational: resolving case reference from the query -> "
                f"{self._describe_case(seed_id) if seed_id is not None else 'none found'}"
            )
        if seed_id is None:
            resolution = "bm25_top1"
            fallback = self.legal_scorer.search(query, 1)
            logs.append(
                f"[AGENT][ACT] relational: no case id in the query, falling back to the "
                f"BM25 top-1 legal case ({self.legal_scorer.scored_documents} scored)"
            )
            if not fallback:
                elapsed = (time.perf_counter() - started) * 1000.0
                return {
                    "ranked": [],
                    "stats": {
                        "corpus_size": float(self.legal_index.N),
                        "candidates_scored": float(self.legal_scorer.scored_documents),
                        "graph_nodes": float(self.graph.node_count()),
                        "graph_edges": float(self.graph.edge_count()),
                        "clusters_probed": 0.0,
                    },
                    "logs": logs
                    + [
                        "[AGENT][OBSERVE] no seed case could be resolved; "
                        "graph path yields nothing"
                    ],
                    "elapsed_ms": elapsed,
                    "nodes": [],
                }
            seed_id = fallback[0][1]

        seed_external = (
            self.legal_index.external_ids[seed_id]
            if 0 <= seed_id < self.legal_index.N
            else str(seed_id)
        )
        logs.append(
            f"[AGENT][ACT] relational: seed case {seed_external} "
            f"(resolution={resolution}), BFS depth={bfs_depth}, "
            f"edge_type={edge_type or 'any'}, direction=both"
        )

        bfs_started = time.perf_counter()
        traversal = self.graph.bfs_traverse(
            seed_external, max_depth=bfs_depth, edge_type=edge_type, direction="both"
        )
        bfs_ms = (time.perf_counter() - bfs_started) * 1000.0

        by_external = self._external_to_id()
        nodes: List[Dict[str, object]] = []
        for node, depth, parent, edge_type in traversal:
            doc_id = by_external.get(node)
            if doc_id is None:
                continue
            nodes.append(
                {
                    "doc_id": doc_id,
                    "external_id": node,
                    "title": self.legal_index.titles[doc_id],
                    "snippet": self.legal_index.snippets[doc_id],
                    "score": None,
                    "depth": depth,
                    "parent": parent,
                    "edge_type": edge_type,
                }
            )
        nodes.sort(key=lambda row: (int(row["depth"]), str(row["external_id"])))
        selected = nodes[:k]
        elapsed = (time.perf_counter() - started) * 1000.0

        stats = {
            "corpus_size": float(self.legal_index.N),
            "candidates_scored": float(len(traversal)),
            "clusters_probed": 0.0,
            "graph_nodes": float(self.graph.node_count()),
            "graph_edges": float(self.graph.edge_count()),
            "bfs_reached": float(len(traversal)),
            "bfs_ms": bfs_ms,
            "seed_doc_id": float(seed_id),
        }
        logs.append(
            f"[AGENT][OBSERVE] BFS reached {len(traversal)} case(s) within depth {bfs_depth} "
            f"in {bfs_ms:.2f} ms; graph has {self.graph.node_count()} nodes / "
            f"{self.graph.edge_count()} edges"
        )
        return {
            "ranked": [],
            "nodes": selected,
            "stats": stats,
            "logs": logs,
            "elapsed_ms": elapsed,
        }

    # -- Observe / Answer --------------------------------------------------

    def observe(
        self,
        path: str,
        payload: Dict[str, object],
        k: int,
        baseline_ranked: Optional[List[Tuple[float, int]]] = None,
    ) -> Tuple[List[Dict[str, object]], List[str]]:
        """Turn a raw payload into result rows and observation logs.

        Args:
            path: The path taken.
            payload: Output of :meth:`act`.
            k: Number of results requested.
            baseline_ranked: Baseline ranking, when comparing paths.

        Returns:
            ``(result_rows, log_lines)``.

        Complexity:
            Time O(k); space O(k).
        """
        logs: List[str] = []
        if path == "relational":
            rows = list(payload.get("nodes", []))  # type: ignore[arg-type]
            logs.append(
                f"[AGENT][OBSERVE] returning {len(rows)} connected case(s) "
                f"including the seed"
            )
            return rows, logs

        ranked: List[Tuple[float, int]] = payload.get("ranked", [])  # type: ignore[assignment]
        index = self.news_index
        rows = []
        for score, doc_id in ranked[:k]:
            rows.append(
                {
                    "doc_id": doc_id,
                    "external_id": index.external_ids[doc_id],
                    "title": index.titles[doc_id],
                    "snippet": index.snippets[doc_id],
                    "score": score,
                    "label": index.labels[doc_id],
                }
            )
        if baseline_ranked is not None and path == "ml":
            baseline_ids = [doc_id for _score, doc_id in baseline_ranked]
            recovered = HashSetTable()
            for doc_id in baseline_ids:
                recovered.put(doc_id, 1)
            overlap = 0
            for row in rows:
                if recovered.contains(row["doc_id"]):
                    overlap += 1
            logs.append(
                f"[AGENT][OBSERVE] recall vs baseline: {overlap}/{len(baseline_ids)} "
                f"baseline results recovered ({100.0 * overlap / max(1, len(baseline_ids)):.1f}%)"
            )
        logs.append(f"[AGENT][OBSERVE] returning {len(rows)} ranked document(s)")
        return rows, logs

    def answer(self, path: str, rows: Sequence[Dict[str, object]], query: str) -> str:
        """Compose the natural-language answer.

        Args:
            path: The path taken.
            rows: Result rows.
            query: Original query.

        Returns:
            One-paragraph answer; honest about an empty result set.

        Complexity:
            Time O(number of rows); space O(same).
        """
        if not rows:
            if path == "relational":
                return (
                    f"No case connected to {query!r} was found within the depth limit. "
                    "Try a case identifier such as 06_1 or a fuller case name."
                )
            return (
                f"No indexed document scored above zero for {query!r}. "
                "Every query term was either unknown to the vocabulary or a stop word."
            )
        if path == "relational":
            seed = rows[0]
            others = [row for row in rows if row.get("depth") not in (0, None)]
            seed_external = str(seed.get("external_id"))
            citing = [row for row in others if str(row.get("parent")) == seed_external]
            unique_classes: HashSetTable = HashSetTable()
            for row in others:
                if row.get("edge_type"):
                    unique_classes.put(str(row.get("edge_type")), 1)
            edge_classes = sorted(unique_classes.keys())
            classes_text = f" (edge types: {', '.join(edge_classes)})" if edge_classes else ""
            return (
                f"Starting from {seed.get('title')!r} "
                f"({seed.get('external_id')}), a breadth-first "
                f"walk of the citation graph reached {len(others)} connected "
                f"case(s) within the "
                f"depth limit{classes_text}; {len(citing)} of them are directly "
                f"connected to the "
                f"seed (depth 1). Results are ordered by hop distance."
            )
        top = rows[0]
        score = float(top.get("score") or 0.0)
        return (
            f"{len(rows)} document(s) matched {query!r}. The strongest is "
            f"{top.get('title')!r} (BM25 {score:.3f}) from the "
            f"{top.get('label') or 'newsgroups'} category."
        )

    # -- Run ---------------------------------------------------------------

    def run(
        self,
        query: str,
        mode: str = "baseline",
        k: int = 10,
        probe_clusters: int = DEFAULT_PROBE_CLUSTERS,
        bfs_depth: int = DEFAULT_BFS_DEPTH,
        compare_baseline: bool = False,
        edge_type: Optional[str] = None,
    ) -> AgentResult:
        """Execute the full Reason -> Act -> Observe loop.

        Args:
            query: Raw query.
            mode: ``"baseline"`` or ``"ml"``.
            k: Number of results.
            probe_clusters: Clusters to probe on the ML path.
            bfs_depth: BFS horizon on the relational path.
            compare_baseline: Also run the baseline so :meth:`observe` can
                report recall.  Costs a second full search.
            edge_type: Optional citation-class filter for the relational BFS.

        Returns:
            An :class:`AgentResult`.

        Complexity:
            Time dominated by the chosen path: O(P log K) for lexical,
            O(V + E) for relational, times a factor of two when
            ``compare_baseline`` is set; space O(D + K).
        """
        started = time.perf_counter()
        intent, reason_logs = self.reason(query, mode)
        act_started = time.perf_counter()
        path, payload = self.act(
            query, intent, mode, k, probe_clusters, bfs_depth, edge_type
        )
        act_ms = (time.perf_counter() - act_started) * 1000.0

        baseline_ranked: Optional[List[Tuple[float, int]]] = None
        baseline_ms = 0.0
        if compare_baseline and path == "ml":
            baseline_started = time.perf_counter()
            baseline_ranked = self.news_scorer.search(query, k)
            baseline_ms = (time.perf_counter() - baseline_started) * 1000.0

        rows, observe_logs = self.observe(path, payload, k, baseline_ranked)
        text = self.answer(path, rows, query)
        answer_log = f"[AGENT][ANSWER] {text}"
        # `total_ms` is the path cost plus the reasoning/observe overhead that
        # would happen anyway.  The comparison baseline is extra work the
        # caller explicitly asked for, so it is reported separately and not
        # folded into `total_ms`; `wall_ms` is the honest end-to-end number.
        total_ms = (time.perf_counter() - act_started - baseline_ms / 1000.0) * 1000.0
        total_ms += (act_started - started) * 1000.0
        wall_ms = (time.perf_counter() - started) * 1000.0

        logs = list(reason_logs) + list(payload.get("logs", []))  # type: ignore[arg-type]
        logs += observe_logs + [answer_log]

        stats: Dict[str, float] = dict(payload.get("stats", {}))  # type: ignore[arg-type]
        if baseline_ranked is not None:
            baseline_ids = HashSetTable()
            for _score, doc_id in baseline_ranked:
                baseline_ids.put(doc_id, 1)
            overlap = 0
            for row in rows:
                if baseline_ids.contains(row["doc_id"]):
                    overlap += 1
            stats["baseline_overlap"] = float(overlap)
            stats["baseline_k"] = float(len(baseline_ids))
            stats["baseline_ms"] = baseline_ms
            stats["speedup"] = baseline_ms / max(1e-9, float(payload.get("elapsed_ms", 1.0)))

        timings = {
            "total": total_ms,
            "wall": wall_ms,
            "act": act_ms,
            "path": float(payload.get("elapsed_ms", 0.0)),  # type: ignore[arg-type]
        }
        for key in ("vectorize_ms", "probe_ms", "score_ms", "bfs_ms"):
            value = stats.get(key)
            if value is not None:
                timings[key.replace("_ms", "")] = float(value)

        if self.verbose:
            for line in logs:
                print(line)

        return AgentResult(query, intent, path, mode, rows, logs, timings, stats, text)

    # -- helpers -----------------------------------------------------------

    def _extract_case_id(self, query: str) -> Optional[int]:
        """Resolve a case reference embedded in a query.

        Scans bracketed citation forms, file-style ids and the whole query
        against the corpus alias table, the longest candidate first so that
        ``"[2006] FCA 1"`` beats the bare ``"1"``.

        Args:
            query: Raw query.

        Returns:
            ``doc_id`` or ``None``.

        Complexity:
            Time O(len(query)^2) worst case for the substring candidates
            (bounded by query length, which is tiny); space O(len(query)).
        """
        corpus = self.legal_corpus
        if corpus is None:
            return None
        candidates: List[str] = []
        # Bracketed citation, e.g. "[2006] FCA 1".
        bracket_open = -1
        for index, char in enumerate(query):
            if char == "[":
                bracket_open = index
            elif char == "]" and bracket_open >= 0:
                candidates.append(query[bracket_open : index + 1])
                bracket_open = -1
        # File-style ids such as 06_1 or case 06_1.
        token: List[str] = []
        tokens: List[str] = []
        for char in query:
            if char.isalnum() or char == "_":
                token.append(char)
            elif token:
                tokens.append("".join(token))
                token = []
        if token:
            tokens.append("".join(token))
        candidates.extend(tokens)
        candidates.append(query)

        ordered = sorted(candidates, key=lambda text: (-len(text), text))
        for candidate in ordered:
            resolved = corpus.resolve_case(candidate)
            if resolved is not None:
                return resolved
        return None

    def _external_to_id(self) -> HashSetTable:
        """Map case external ids to doc ids, built once and reused.

        The legal index is immutable after construction, so the mapping is
        cached on first use.  Without the cache every relational query paid
        3,890 hash inserts against a 12-16 ms BFS that the map exists to serve.
        ``bulk_put`` sizes the table once, which measures 4.7 ms instead of the
        13.8 ms an insert-at-a-time loop costs (each resize rehashes).

        Returns:
            A :class:`HashSetTable` from external id to ``doc_id``.

        Complexity:
            Time O(n) expected on the first call, O(1) afterwards; space O(n),
            retained for the agent's lifetime.
        """
        if self._external_index is not None:
            return self._external_index
        mapping: HashSetTable = HashSetTable()
        mapping.bulk_put(
            (external_id, doc_id)
            for doc_id, external_id in enumerate(self.legal_index.external_ids)
        )
        self._external_index = mapping
        return mapping

    def _describe_case(self, doc_id: Optional[int]) -> str:
        """Human-readable label for a legal doc id.

        Args:
            doc_id: Legal document identifier.

        Returns:
            ``"06_1 (name)"`` style text.

        Complexity: O(1).
        """
        if doc_id is None:
            return "none"
        title = self.legal_index.titles[doc_id]
        return f"{self.legal_index.external_ids[doc_id]} ({title[:70]})"

    @staticmethod
    def _pct(part: float, whole: float) -> str:
        """Format ``part/whole`` as a percentage string.

        Args:
            part: Numerator.
            whole: Denominator.

        Returns:
            ``"12.3%"``.

        Complexity: O(1).
        """
        if whole <= 0:
            return "0.0%"
        return f"{100.0 * part / whole:.1f}%"


__all__ = ["ReActAgent", "AgentResult", "is_relational"]
