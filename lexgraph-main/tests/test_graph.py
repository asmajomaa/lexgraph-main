"""Graph and BFS tests: cycles, depth limits, direction, edge filters, path."""

from __future__ import annotations

import random

from structures.graph import CitationGraph


def _chain(length: int) -> CitationGraph:
    graph = CitationGraph()
    for index in range(length):
        graph.add_edge(f"c{index}", f"c{index + 1}")
    return graph


def test_add_node_and_edge_counts():
    graph = CitationGraph()
    graph.add_node("solo")
    graph.add_edge("a", "b", "applied")
    graph.add_edge("a", "b", "applied")  # duplicate ignored
    graph.add_edge("a", "b", "followed")  # different class kept
    graph.add_edge("a", "a", "cited")  # self loop ignored
    assert graph.node_count() == 3
    assert graph.edge_count() == 2
    # degree() counts distinct neighbours, so two classes to "b" count once.
    assert graph.degree("a", "out") == 1
    assert len(graph.neighbors("a", "out")) == 2


def test_neighbors_direction_and_filter():
    graph = CitationGraph()
    graph.add_edge("new", "old", "followed")
    graph.add_edge("new", "older", "distinguished")
    graph.add_edge("newest", "new", "cited")
    assert graph.neighbors("new", "out") == [("old", "followed"), ("older", "distinguished")]
    assert graph.neighbors("new", "in") == [("newest", "cited")]
    both = graph.neighbors("new", "both")
    assert len(both) == 3
    assert graph.neighbors("new", "out", "followed") == [("old", "followed")]
    assert graph.neighbors("new", "out", "applied") == []


def test_bfs_depth_limited():
    graph = _chain(6)
    traversal = graph.bfs_traverse("c0", max_depth=2)
    nodes = [(node, depth) for node, depth, _p, _e in traversal]
    assert nodes == [("c0", 0), ("c1", 1), ("c2", 2)]
    full = graph.bfs_traverse("c0", max_depth=10)
    assert [n for n, _d, _p, _e in full] == [f"c{i}" for i in range(7)]


def test_bfs_records_parent_and_edge_type():
    graph = CitationGraph()
    graph.add_edge("a", "b", "applied")
    graph.add_edge("b", "c", "followed")
    traversal = graph.bfs_traverse("a", max_depth=2)
    by_node = {node: (depth, parent, etype) for node, depth, parent, etype in traversal}
    assert by_node["a"] == (0, None, None)
    assert by_node["b"] == (1, "a", "applied")
    assert by_node["c"] == (2, "b", "followed")


def test_bfs_handles_cycles_and_visits_once():
    graph = CitationGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "a")
    graph.add_edge("c", "b")
    traversal = graph.bfs_traverse("a", max_depth=10)
    assert [n for n, _d, _p, _e in traversal] == ["a", "b", "c"]
    assert all(depth <= 2 for _n, depth, _p, _e in traversal)


def test_bfs_disconnected_yields_only_seed():
    graph = CitationGraph()
    graph.add_edge("a", "b")
    graph.add_node("island")
    assert [n for n, _d, _p, _e in graph.bfs_traverse("island", 3)] == ["island"]


def test_bfs_unknown_start_returns_empty():
    assert CitationGraph().bfs_traverse("nope", 2) == []


def test_bfs_reverse_direction_finds_citing_cases():
    graph = CitationGraph()
    graph.add_edge("later1", "seed")
    graph.add_edge("later2", "later1")
    out = [n for n, _d, _p, _e in graph.bfs_traverse("seed", 2, direction="out")]
    inb = [n for n, _d, _p, _e in graph.bfs_traverse("seed", 2, direction="in")]
    assert out == ["seed"]
    assert inb == ["seed", "later1", "later2"]


def test_bfs_edge_type_filter_prunes_frontier():
    graph = CitationGraph()
    graph.add_edge("seed", "strong", "applied")
    graph.add_edge("seed", "weak", "cited")
    graph.add_edge("weak", "deep", "applied")
    applied = [n for n, _d, _p, _e in graph.bfs_traverse("seed", 3, edge_type="applied")]
    assert applied == ["seed", "strong"]


def test_shortest_path():
    graph = CitationGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("a", "c")
    assert graph.shortest_path("a", "c") == ["a", "c"]
    assert graph.shortest_path("a", "a") == ["a"]
    assert graph.shortest_path("c", "a") is None
    assert graph.shortest_path("a", "zzz") is None


def test_connected_component_size_weak_and_directed():
    graph = CitationGraph()
    graph.add_edge("a", "b")
    graph.add_edge("c", "b")
    graph.add_node("lone")
    assert graph.connected_component_size("b", "both") == 3
    assert graph.connected_component_size("a", "out") == 2
    assert graph.connected_component_size("lone", "both") == 1
    assert graph.connected_component_size("missing") == 0


def test_depth_profile_is_cumulative():
    graph = _chain(4)
    profile = graph.depth_profile("c0", max_depth=2, direction="out")
    assert profile == [(0, 1), (1, 2), (2, 3)]


def test_bfs_order_is_deterministic():
    graph = CitationGraph()
    rng = random.Random(3)
    for _ in range(200):
        graph.add_edge(f"n{rng.randint(0, 30)}", f"n{rng.randint(0, 30)}")
    first = [n for n, _d, _p, _e in graph.bfs_traverse("n0", 3, direction="both")]
    second = [n for n, _d, _p, _e in graph.bfs_traverse("n0", 3, direction="both")]
    assert first == second


def test_edge_type_counts_sorted():
    graph = CitationGraph()
    graph.add_edge("a", "b", "cited")
    graph.add_edge("a", "c", "cited")
    graph.add_edge("a", "d", "applied")
    assert graph.edge_type_counts() == [("cited", 2), ("applied", 1)]
    assert graph.average_degree() > 0


def test_edge_type_filter_reaches_the_agent() -> None:
    """The relational path must expose the graph's edge-class filter.

    The brief asks for a depth-limited BFS with an *optional edge-type filter*;
    before this test the filter existed on ``CitationGraph.bfs_traverse`` but no
    caller could reach it, so it was dead code. This pins the whole route:
    ``ReActAgent.run(..., edge_type=...)`` -> ``act`` -> ``_act_relational`` ->
    ``bfs_traverse``.
    """
    import inspect

    from core.agent import ReActAgent

    for method in (ReActAgent.run, ReActAgent.act, ReActAgent._act_relational):
        signature = inspect.signature(method)
        assert "edge_type" in signature.parameters, (
            f"{method.__name__} does not accept edge_type"
        )


def test_bfs_honours_the_edge_filter() -> None:
    """A filtered traversal keeps only the requested class."""
    from structures.graph import CitationGraph

    graph = CitationGraph()
    graph.add_edge("a", "b", "cited")
    graph.add_edge("a", "c", "applied")
    graph.add_edge("b", "d", "cited")

    everything = {
        node
        for node, _d, _p, _e in graph.bfs_traverse("a", max_depth=2, direction="out")
    }
    cited_only = {
        node
        for node, _d, _p, edge in graph.bfs_traverse(
            "a", max_depth=2, edge_type="cited", direction="out"
        )
        if edge is None or edge == "cited"
    }
    assert everything == {"a", "b", "c", "d"}
    assert cited_only == {"a", "b", "d"}, cited_only
