# 1. Objective of the project

LEXGRAPH-RAG is a hybrid, agentic information-retrieval engine over two real
document collections: the **20 Newsgroups** corpus (Mitchell, 1997) and the
**Legal Case Reports** corpus (Galgani, 2010). The project exists to show, in
one runnable system, that three classical fields fit together rather than
competing:

| field | what the project contributes | where it lives |
|---|---|---|
| **Information retrieval** | an inverted index and an Okapi BM25 scorer, both written from scratch | `structures/hash_table.py`, `core/inverted_index.py`, `core/bm25.py` |
| **Machine learning** | TF-IDF vectorisation and K-Means used as a *cluster-pruning* accelerator, not as a topic model to be admired | `ml/vectorizer.py`, `ml/kmeans.py` |
| **Graph theory** | a typed, directed citation graph with BFS, shortest path and component queries | `structures/graph.py` |

The graded constraint is as much the point as the retrieval quality: **the
engine builds its own hash table, its own binary heap, its own graph, its own
vectoriser and its own K-Means**, and imports no `dict`-as-index, no `heapq`, no
`networkx`, no `scikit-learn` and no `numpy` anywhere in `structures/`, `core/`
or `ml/`. The test suite enforces that mechanically: `tests/test_purity.py`
parses every engine module and fails on a forbidden import, *and* fails on any
empty `{}`/`set()`, comprehension or zero-argument `dict()`/`set()` built inside
engine code -- the shapes a lookup table takes. The claim is therefore
verifiable rather than asserted.

## 1.1 What "agentic" means here

The agent is a **rule-based ReAct loop** (`core/agent.py`), not an LLM. Each
query goes through four logged steps:

```text
[AGENT][REASON]   classify the intent: RELATIONAL or LEXICAL
[AGENT][ACT]      dispatch to one of three retrieval paths
[AGENT][OBSERVE]  report what the path did (documents scored, hops, timings)
[AGENT][ANSWER]   one sentence stating the strongest result and why
```

That gives three paths the report can measure separately:

1. **Relational** — the query mentions a citation relation ("which cases cite
   06_584") → resolve the case id, then BFS the citation graph with a depth
   limit and an optional edge-class filter, in both directions.
2. **Lexical baseline** — tokenise the query, walk the inverted index, score
   every matching document with BM25, keep the top K with a bounded min-heap.
3. **Lexical ML** — vectorise the query, ask K-Means for the nearest
   cluster(s), and score only the documents in them. The saving is measured,
   and so is the recall lost by pruning.

Every number in this report comes from a real run of the code. The commands
that produced them are printed beside the tables, and
`report/tables/summary.json` is the machine-readable record.

## 1.2 Datasets actually used

| dataset | source on disk | documents indexed | labels used |
|---|---|---|---|
| 20 Newsgroups | `data/raw/20news-bydate-train` + `-test` (18,846 files: 11,314 + 7,532; `bydate` layout) | **18,719** after dropping posts whose cleaned body is under 20 characters | 20 newsgroup names, used **only** to score the clustering (purity / NMI / ARI) |
| Legal Case Reports | `data/raw/corpus` (3,890 XML files, one case per file) | **3,890** | 17 citation classes found in the catchphrase markup, used as typed edge labels |

The README repeats this mapping, because the brief asks for it explicitly:
 *all retrieval and clustering numbers in this report use the real corpora, not
 the synthetic fallbacks.* The synthetic generators (`data/synthetic.py`) exist
so the project still runs on a machine without the datasets, and the loaders
print which source they used; no result in this report was produced from them.

## 1.3 What the deliverable does

```bash
python main.py --build                      # parse both corpora, build every index, cache
python main.py --query "..." --mode baseline --k 10
python main.py --query "..." --mode ml --compare      # also reports recall vs baseline
python main.py --query "which cases cite 06_584"      # routes to the relational path
python main.py --interactive                # REPL with :mode :k :probe :depth :build
python main.py --serve                      # workbench UI + JSON API on 127.0.0.1:8000
python -m benchmarks.run_all                # every figure and table in this report
pytest                                      # the full suite
```

The build takes **193.00 s** end to end on the machine used for every
measurement (Windows 11, CPython 3.10), of which K-Means is 93.74 s, the legal
index 48.54 s and corpus loading 16.13 s. The full build transcript is in
`report/tables/transcript_build.txt` and is reproduced in section 7.
