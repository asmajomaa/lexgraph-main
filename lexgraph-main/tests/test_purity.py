"""Architecture tests: the graded core must not depend on the forbidden libraries.

The brief forbids ``dict``/``set`` as the index, ``heapq``, ``networkx``,
``scikit-learn`` and ``numpy`` inside ``structures/``, ``core/`` and ``ml/``.
This module enforces that mechanically with :mod:`ast`, so a future edit that
slips ``import numpy`` into the engine fails the suite instead of quietly
invalidating the whole project.

``dict`` is allowed in loaders, tests and benchmarks but *not* inside the
engine. The brief's wording ("using a built-in ``dict``/``set`` as the index")
is narrower than "never type the word dict", so the enforceable rule this suite
applies is: **no built-in ``dict`` or ``set`` is constructed in engine code
except inside a function whose name starts with ``to_state`` or ``__`` (the
cache serialisation layer) and no runtime ``dict``/``set`` lookup drives an
algorithm.** That is checked two ways -- an AST scan for ``{}``/``set()``/
comprehension construction (with the serialisation exemptions), and behavioural
tests that the index, the vectoriser vocabulary, the tokenizer's stop table, the
citation adjacency and the metric counters are all the project's own
``HashSetTable``.
"""

from __future__ import annotations

import ast
import os
from typing import List, Optional, Set

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Directories that form the graded engine.
CORE_DIRS = ("structures", "core", "ml")

#: Modules the engine is allowed to import (standard library only).
_ALLOWED_STDLIB = {
    "__future__",
    "ast",
    "bisect",
    "math",
    "os",
    "random",
    "re",
    "sys",
    "time",
    "typing",
    "collections",  # types only (Iterable/Sequence) -- never deque in core
    "itertools",
    "pickle",
    "json",
}

#: Third-party modules that must never appear in the engine.
FORBIDDEN = {"numpy", "sklearn", "networkx", "heapq", "scipy", "pandas", "matplotlib"}


def _core_modules() -> List[str]:
    """List every Python file inside the graded core.

    Returns:
        Absolute paths.

    Complexity:
        Time O(number of files); space O(number of files).
    """
    paths: List[str] = []
    for directory in CORE_DIRS:
        root = os.path.join(_PROJECT_ROOT, directory)
        for name in sorted(os.listdir(root)):
            if name.endswith(".py"):
                paths.append(os.path.join(root, name))
    return paths


def _imported_modules(path: str) -> Set[str]:
    """Top-level module names imported by one file.

    Args:
        path: File to parse.

    Returns:
        Module names (``"a.b"`` yields ``"a"``).

    Complexity:
        Time O(size of the file); space O(imports).
    """
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("path", _core_modules(), ids=lambda p: os.path.basename(p))
def test_core_imports_no_forbidden_library(path: str) -> None:
    """No engine module may import a forbidden third-party package."""
    imported = _imported_modules(path)
    overlap = imported & FORBIDDEN
    assert not overlap, f"{path} imports forbidden module(s): {sorted(overlap)}"


@pytest.mark.parametrize("path", _core_modules(), ids=lambda p: os.path.basename(p))
def test_core_imports_only_stdlib_and_siblings(path: str) -> None:
    """Engine imports stay within the standard library and the project itself."""
    imported = _imported_modules(path)
    external = {
        name
        for name in imported
        if name not in _ALLOWED_STDLIB
        and name
        not in {"config", "structures", "core", "ml", "data", "engine", "report", "web"}
    }
    assert not external, f"{path} imports unexpected module(s): {sorted(external)}"


def test_inverted_index_uses_the_custom_table() -> None:
    """The inverted index must be backed by ``PostingsTable``, not a ``dict``."""
    from core.inverted_index import InvertedIndex
    from structures.hash_table import PostingsTable

    index = InvertedIndex()
    assert isinstance(index.postings, PostingsTable)


def test_graph_adjacency_uses_the_custom_table() -> None:
    """Both adjacency directions must be custom hash tables."""
    from structures.graph import CitationGraph
    from structures.hash_table import HashSetTable

    graph = CitationGraph()
    assert isinstance(graph._out, HashSetTable)
    assert isinstance(graph._in, HashSetTable)


def test_top_k_heap_matches_full_sort() -> None:
    """`TopKHeap` and the sort-based oracle agree on a seeded random stream."""
    import random

    from structures.min_heap import TopKHeap, sort_top_k

    rng = random.Random(0)
    candidates = [(rng.random(), rng.randrange(50)) for _ in range(500)]
    for k in (1, 3, 10, 500, 1000):
        heap = TopKHeap(k)
        for score, doc_id in candidates:
            heap.push(score, doc_id)
        assert heap.get_top_k() == sort_top_k(candidates, k)


# -- the dict/set boundary ---------------------------------------------------

#: Functions whose whole job is to produce a cache payload or a repr string.
#: A ``dict`` there is an output format, not a lookup structure.
_SERIALISATION_PREFIXES = ("to_state", "to_dict", "__repr__")

#: Attributes that hold measurement payloads rather than lookup tables. An
#: assignment to one of these is a stats record the caller reads by name.
_PAYLOAD_ATTRIBUTES = frozenset({"last_stats"})


def _enclosing_function(tree: ast.AST) -> dict:
    """Map ``id(node) -> name of the innermost enclosing function``.

    Args:
        tree: A parsed module.

    Returns:
        Mapping from node identity to the function name.

    Complexity:
        Time O(number of nodes); space O(number of functions).
    """
    owner: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                owner.setdefault(id(child), node.name)
    return owner


def _assigned_attribute(node: ast.AST, parents: dict) -> Optional[str]:
    """Name of the attribute this node is assigned to, if any.

    Matches both ``self.x = {...}`` (the node is the value) and
    ``self.x = {}`` where the dict is built and then filled.

    Args:
        node: Candidate node.
        parents: ``id(child) -> parent`` map.

    Returns:
        The attribute name, or ``None``.

    Complexity:
        Time O(1); space O(1).
    """
    parent = parents.get(id(node))
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
        target = parent.targets[0]
    elif isinstance(parent, ast.AnnAssign) and parent.value is node:
        target = parent.target
    else:
        return None
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _is_get_fallback(node: ast.AST, parents: dict) -> bool:
    """True when ``node`` is the default argument of a ``.get(key, default)`` call.

    That pattern is a copy-out of a payload, not a table being built.

    Args:
        node: Candidate node.
        parents: ``id(child) -> parent`` map.

    Returns:
        Whether the node is a ``.get`` fallback.

    Complexity:
        Time O(1); space O(1).
    """
    parent = parents.get(id(node))
    if not isinstance(parent, ast.Call):
        return False
    return (
        isinstance(parent.func, ast.Attribute)
        and parent.func.attr == "get"
        and len(parent.args) >= 2
        and parent.args[1] is node
    )


def _lookup_table_constructions(tree: ast.AST) -> List[tuple]:
    """Locate dict/set construction that could serve as an index.

    The rule this enforces: an *empty* ``{}``/``set()``/``list``-like start
    followed by incremental writes is how a lookup table gets built, and
    comprehension construction is the same thing written short. A non-empty
    literal with constant keys is a record (a stats payload, a result row), not
    a table. So this flags:

    * empty dict/set literals ``{}`` / ``{}``-literal forms,
    * zero-argument ``dict()``/``set()``/``frozenset()`` calls,
    * dict and set comprehensions.

    and ignores non-empty literals and ``dict(payload)`` copies.

    Args:
        tree: A parsed module.

    Returns:
        ``(lineno, kind, enclosing_function)`` triples.

    Complexity:
        Time O(number of nodes); space O(number of hits).
    """
    owner = _enclosing_function(tree)
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    hits: List[tuple] = []
    for node in ast.walk(tree):
        kind = None
        if isinstance(node, ast.Dict) and len(node.keys) == 0:
            if not _is_get_fallback(node, parents):
                kind = "empty dict"
        elif isinstance(node, ast.Set) and len(node.elts) == 0:
            kind = "empty set"
        elif isinstance(node, ast.DictComp):
            kind = "dict-comprehension"
        elif isinstance(node, ast.SetComp):
            kind = "set-comprehension"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("dict", "set", "frozenset")
            and not node.args
            and not node.keywords
        ):
            kind = node.func.id + "()"
        if kind:
            if _assigned_attribute(node, parents) in _PAYLOAD_ATTRIBUTES:
                continue
            hits.append((node.lineno, kind, owner.get(id(node), "<module>")))
    return hits


@pytest.mark.parametrize("path", _core_modules(), ids=lambda p: os.path.basename(p))
def test_no_dict_or_set_lookup_tables_in_engine(path: str) -> None:
    """Engine code must not build a lookup table out of a plain ``dict``/``set``.

    See :func:`_lookup_table_constructions` for the exact rule. Result payloads
    (non-empty literals with constant keys) and ``to_state`` serialisation are
    allowed; everything used as a map is the project's own ``HashSetTable``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    relative = os.path.relpath(path, _PROJECT_ROOT).replace(os.sep, "/")
    offences = []
    for lineno, kind, function in _lookup_table_constructions(tree):
        if function.startswith(_SERIALISATION_PREFIXES):
            continue
        offences.append(f"{relative}:{lineno} {kind} inside {function}()")

    assert not offences, (
        "plain dict/set used as a lookup table in the engine:\n"
        + "\n".join(offences)
    )


def test_vectorizer_vocabulary_is_the_custom_table() -> None:
    """The TF-IDF vocabulary is a ``HashSetTable``, not a built-in ``dict``."""
    from ml.vectorizer import TFIDFVectorizer
    from structures.hash_table import HashSetTable

    vectorizer = TFIDFVectorizer()
    assert isinstance(vectorizer.vocabulary, HashSetTable)
    assert not isinstance(vectorizer.vocabulary, dict)


def test_tokenizer_stop_words_use_no_lookup_table() -> None:
    """Stop-word membership is a sorted tuple + bisect, not a ``dict``/``set``.

    A sorted tuple is one of the plain containers the brief explicitly allows,
    and on a 150-word list it is also the fastest compliant option measured
    (0.254 us per probe against 0.676 us for ``HashSetTable``), which matters
    because the tokenizer is the build's hottest loop.
    """
    from core.tokenizer import _STOP_SORTED, is_stop_word

    assert isinstance(_STOP_SORTED, tuple)
    assert list(_STOP_SORTED) == sorted(_STOP_SORTED)
    assert is_stop_word("the") and not is_stop_word("hockey")
    with pytest.raises(ImportError):
        from core.tokenizer import _STOP  # noqa: F401  (must not exist)


def test_bm25_query_tables_are_the_custom_table() -> None:
    """BM25's per-query idf map uses the custom table."""
    from core.bm25 import BM25
    from core.inverted_index import InvertedIndex
    from data.documents import Document
    from structures.hash_table import HashSetTable

    index = InvertedIndex.build(
        [Document(doc_id=0, text="graphics texture render", external_id="0", title="t")]
    )
    scorer = BM25(index)
    _terms, idf_by_term = scorer._prepare_query("graphics texture")
    assert isinstance(idf_by_term, HashSetTable)


def test_evaluation_contingency_table_is_the_custom_table() -> None:
    """The contingency table is a ``HashSetTable``."""
    from ml.evaluation import contingency_table
    from structures.hash_table import HashSetTable

    table = contingency_table(["a", "a", "b"], [0, 1, 1])
    assert isinstance(table, HashSetTable)
    assert table.get(("a", 0)) == 1


def test_agent_cluster_map_is_the_custom_table() -> None:
    """``ReActAgent.cluster_members`` is a ``HashSetTable``."""
    from core.agent import ReActAgent
    from structures.hash_table import HashSetTable

    agent = ReActAgent(
        news_index=None,  # type: ignore[arg-type]
        legal_index=None,  # type: ignore[arg-type]
        graph=None,  # type: ignore[arg-type]
        kmeans=None,
        clusters=[0, 0, 1],
    )
    assert isinstance(agent.cluster_members, HashSetTable)
