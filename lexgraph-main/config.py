"""Central configuration for LEXGRAPH-RAG.

Every tunable constant used by the engine lives here so that no magic number
is buried inside an algorithm.  Values are grouped by the subsystem that
consumes them.

The module is dependency-free: it imports nothing from the rest of the project
and nothing outside the Python standard library's ``os`` and ``pathlib``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PACKAGE_ROOT / "data" / "raw"
CACHE_DIR: Path = PACKAGE_ROOT / "cache"
REPORT_DIR: Path = PACKAGE_ROOT / "report"
FIGURES_DIR: Path = REPORT_DIR / "figures"
TABLES_DIR: Path = REPORT_DIR / "tables"

#: Environment variable that may point at a *pre-processed* corpus root.
#: When it is set, the loaders read from there instead of parsing raw files.
ENV_DATA_ROOT: str = "LEXGRAPH_DATA_ROOT"

# ---------------------------------------------------------------------------
# Hash table (structures/hash_table.py)
# ---------------------------------------------------------------------------

#: Rolling-hash multiplier.  31 is the classic choice for lower-case ASCII
#: vocabulary: it is prime, small, and ``31 * x`` is computed as
#: ``(x << 5) - x`` so the inner loop stays cheap on any CPU.
HASH_BASE: int = 31

#: Hash modulus.  A prime close to 2**61 - 1 keeps the polynomial hash within
#: a single machine word on 64-bit platforms (``2**61 < 2**63``), so Python's
#: arbitrary-precision fallback is never triggered and each word costs O(1).
HASH_MODULUS: int = 2_305_843_009_213_693_951  # 2**61 - 1 (Mersenne prime)

#: Initial bucket count.  Must be a power of two so that the cheap
#: ``hash & (size - 1)`` reduction in ``_bucket_index`` is correct.
HASH_INITIAL_CAPACITY: int = 16

#: Resize (double) once ``size / capacity`` exceeds this value.  0.75 is the
#: textbook sweet spot: it keeps average chain length near 1.5 probes while
#: leaving enough head-room that resizes are rare (amortised O(1)).
HASH_LOAD_FACTOR_MAX: float = 0.75

# ---------------------------------------------------------------------------
# Tokenizer (core/tokenizer.py)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Read an integer override from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or malformed.

    Returns:
        The override, or ``default``.

    Complexity:
        O(1) time, O(1) space.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


#: Tokens shorter than this are dropped (single characters carry little signal
#: in English while inflating the vocabulary by an order of magnitude).
MIN_TOKEN_LENGTH: int = 2

#: Maximum bytes of a newsgroup post fed to the tokenizer.  20 Newsgroups has
#: a few pathological posts of several hundred kilobytes whose tail adds no
#: ranking signal; capping them bounds tokenizer time per document.
MAX_DOCUMENT_CHARS: int = 20_000

# ---------------------------------------------------------------------------
# BM25 (core/bm25.py)
# ---------------------------------------------------------------------------

#: Term-frequency saturation.  1.2-2.0 is the published range; 1.5 is the
#: middle of Robertsson & Zaragoza's recommended band.
BM25_K1: float = 1.5

#: Length normalisation strength.  0.75 is the value used in essentially all
#: modern BM25 deployments.
BM25_B: float = 0.75

# ---------------------------------------------------------------------------
# Vectorizer / K-Means (ml/)
# ---------------------------------------------------------------------------

#: Size of the TF-IDF vocabulary handed to K-Means.  5000 keeps the dense
#: centroid matrix at 5000 floats per cluster -- 20 clusters x 5000 x 8 bytes
#: is 800 kB, so the whole model fits comfortably in cache-friendly memory.
VECTORIZER_VOCAB_SIZE: int = 5000

#: Terms occurring in fewer than this many documents are discarded as noise.
VECTORIZER_MIN_DF: int = 2

#: Terms occurring in more than this fraction of documents are discarded as
#: corpus-wide stop words.
VECTORIZER_MAX_DF_RATIO: float = 0.5

#: Number of clusters.  20 matches the number of newsgroup categories, which
#: makes the labelling experiment directly comparable to the ground truth.
KMEANS_K: int = 20

#: Hard iteration ceiling for Lloyd's algorithm.
KMEANS_MAX_ITER: int = 60

#: Centre-shift threshold used as the second convergence criterion.
KMEANS_TOL: float = 1e-4

#: Fixed seed for k-means++ so every reported number is reproducible.
KMEANS_SEED: int = 42

#: Number of trial points drawn per k-means++ seeding step (the "greedy"
#: variant, ``O(log k)`` approximation guarantee with a constant of 2+).
KMEANS_PP_TRIALS: int = 2

# ---------------------------------------------------------------------------
# Agent (core/agent.py)
# ---------------------------------------------------------------------------

#: Default number of K-Means clusters probed for the ML path.  Probing 1 is
#: fastest but loses recall; probing all ``k`` degenerates to the baseline.
DEFAULT_PROBE_CLUSTERS: int = 2

#: Default BFS depth for the relational path.
DEFAULT_BFS_DEPTH: int = 2

#: Words whose presence routes a query to the citation graph.
RELATIONAL_KEYWORDS: tuple[str, ...] = (
    "cite",
    "cites",
    "cited",
    "citing",
    "citation",
    "citations",
    "reference",
    "references",
    "referenced",
    "followed",
    "applied",
    "distinguished",
    "precedent",
    "precedents",
)

# ---------------------------------------------------------------------------
# Evaluation / benchmarks
# ---------------------------------------------------------------------------

#: Number of queries sampled for the ML-vs-baseline recall study.
EVAL_QUERY_COUNT: int = 100

#: Top-K used when the user does not override it.
DEFAULT_TOP_K: int = 10

#: Repetitions per measurement point.  Five runs give a usable standard
#: deviation without making a full benchmark suite unbearable in pure Python.
BENCH_REPEATS: int = 5
