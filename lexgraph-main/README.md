# LEXGRAPH-RAG

A hybrid, agentic information-retrieval engine over two real corpora — the
**20 Newsgroups** dataset and the **Legal Case Reports** dataset — built to show
three fields in one system:

* **Information retrieval** — a from-scratch inverted index and Okapi BM25.
* **Machine learning** — TF-IDF vectorisation and K-Means used as a *cluster
  pruning* accelerator, measured honestly including its recall cost.
* **Graph theory** — a typed, directed citation graph with BFS, shortest paths
  and component queries.

The engine uses **no `dict`/`set` as a lookup structure, and imports no
`heapq`, `networkx`, `scikit-learn` or `numpy`**, in `structures/`, `core/` or
`ml/`. `tests/test_purity.py` parses each module's AST, fails on a forbidden
import, and fails on any empty `{}`/`set()`, comprehension or zero-argument
`dict()`/`set()` built inside engine code, so the constraint is enforced rather
than promised.

---

## Quick start

```bash
python -m pip install -r requirements.txt

python main.py --build                                  # ~194 s, once
python main.py --query "hockey playoffs goalie" --k 5
python main.py --query "hockey playoffs goalie" --mode ml --compare
python main.py --query "which cases cite 06_584"
python main.py --interactive
python main.py --serve                                  # workbench UI at http://127.0.0.1:8000
python -m benchmarks.run_all                            # every figure and table
pytest                                                  # 232 tests
```

Python 3.10+ is required (`str | None` style typing is avoided, so 3.9 may work,
but only 3.10 was verified).

---

## Datasets: which data produced which results

The loaders print the source they used. **Every number in the report was
produced from the real corpora, not from the synthetic fallbacks.**

| dataset | where it must live | expected layout | used for |
|---|---|---|---|
| 20 Newsgroups (Mitchell, 1997) | `data/raw/20news-bydate-train/` and `data/raw/20news-bydate-test/` (or `20news-18828/`) | one directory per newsgroup, one file per post | all lexical retrieval and all clustering numbers (18,719 documents, 20 categories) |
| Legal Case Reports (Galgani, 2010) | `data/raw/corpus/` | one XML file per case in a nested year directory tree | all relational/graph numbers (3,890 cases, 3,077 citation edges) |

* The 20 Newsgroups path can also be set with `LEXGRAPH_NEWSGROUPS`; the legal
  corpus is discovered under `data/raw` (`corpus/`, or `legal/`).
* Body cleaning for newsgroups strips quoted lines (`> ...`) and everything
  after the `-- ` signature delimiter, per the brief.
* Legal documents are parsed with regular expressions, **not** ElementTree: the
  corpus XML is not well-formed (`<catchphrase "id=c0">`, unquoted attribute
  values). Index text = catchphrases (boosted ×3) + body sentences; edges are
  case → cited case typed by the catchphrase class.
* If a dataset is missing, `data/synthetic.py` generates a small stand-in so the
  project still runs. The loader announces `source: synthetic`, and no reported
  result comes from it.

---

## Command line

| command | what it does |
|---|---|
| `python main.py --build` | parse both corpora, build every index, fit K-Means, write `cache/engine.pkl` |
| `python main.py --query "..." --mode baseline --k 10` | BM25 over the full corpus |
| `python main.py --query "..." --mode ml --probe-clusters 2` | K-Means pruning, then BM25 |
| `python main.py --query "..." --compare` | also run the baseline and report recall |
| `python main.py --query "which cases cite 06_584"` | routes to the relational path automatically |
| `python main.py --explain "hockey playoffs goalie" --doc 3870` | per-term BM25 breakdown |
| `python main.py --interactive` | REPL with `:mode`, `:k`, `:probe`, `:depth`, `:build`, `:quit` |
| `python main.py --serve --port 8000` | workbench web UI + JSON API (8000 is the default) |

Every query prints the ReAct trace:

```text
[AGENT][REASON] intent=LEXICAL: no relational keyword (mode=ml) -> lexical path
[AGENT][ACT] lexical/ml: vectorised query (nnz=3), probing 2 cluster(s) [2, 1] in 0.05 ms
[AGENT][ACT] candidate pool = 5597 of 18719 documents (29.9%); BM25 over candidates in 1.09 ms
[AGENT][OBSERVE] scored 413 documents in 1.21 ms
[AGENT][OBSERVE] recall vs baseline: 5/5 baseline results recovered (100.0%)
[AGENT][ANSWER] 5 document(s) matched 'hockey playoffs goalie'. ...
```

---

## Cache format

`cache/engine.pkl` is a **single pickle** (about 219 MB) containing only plain
Python values — lists, tuples, strings, numbers and dicts of those. Every custom
structure is serialised through a `to_state()` method and rebuilt with
`from_state()`, so the file holds no class definitions and cannot execute code
when loaded.

A **fingerprint** is stored beside the payload:

```python
{'version': 2, 'news': 18719, 'legal': 3890, 'vocab': 5000,
 'min_df': 2, 'k': 20, 'max_iter': 60, 'seed': 42, 'tokenizer': 'default-v1'}
```

It is compared with the current `config.py` on load, so changing a build
parameter invalidates the cache instead of silently serving a stale model. A
schema change requires bumping `CACHE_VERSION` in `engine.py`.

Pickle was chosen over a hand-written text format because the payload is 1.4 M
postings plus 18,719 documents; a text format would triple the size and add a
parser with its own bugs, for a file that is machine-local and regenerated by
`--build`. The trade-off is that the cache is not portable across Python
versions — documented, and fixed by rebuilding.

---

## Project layout

```text
lexgraph_rag/
├── config.py                 every tunable, with the reason for its value
├── engine.py                 assembly, pickle cache, fingerprint validation
├── main.py                   CLI: --build --query --interactive --serve --explain
├── data/
│   ├── documents.py          the Document record
│   ├── load_newsgroups.py    20 Newsgroups loader (bydate / 18828 / synthetic)
│   ├── load_legal_cases.py   Legal Case Reports XML loader + citation records
│   └── synthetic.py          offline fallbacks
├── structures/
│   ├── hash_table.py         polynomial hash, chaining, PostingsTable
│   ├── min_heap.py           binary heap + bounded TopKHeap
│   ├── graph.py              typed directed citation graph + ring-buffer BFS
│   └── sparse_vector.py      sorted parallel arrays, dot / cosine / distance
├── core/
│   ├── tokenizer.py          hand-written tokeniser + light Porter stemmer
│   ├── inverted_index.py     postings, doc lengths, RestrictedView, ClusterFilter
│   ├── bm25.py               OKapi BM25 + per-term explanation
│   └── agent.py              rule-based ReAct agent, three paths
├── ml/
│   ├── vectorizer.py         TF-IDF, top-M by df, smoothed idf, L2
│   ├── kmeans.py             k-means++ seeding, sparse Lloyd, predict_top_n
│   └── evaluation.py         purity, NMI, ARI, confusion matrix (from scratch)
├── benchmarks/
│   ├── bench_scalability.py  query time vs corpus size
│   ├── bench_query_time.py   baseline vs ML-accelerated, Recall@K
│   ├── bench_kmeans.py       inertia curve, PCA scatter, confusion heatmap
│   ├── bench_graph.py        graph statistics, BFS scaling, depth profiles
│   ├── compare_existing.py   vs dict / heapq / rank_bm25 / sklearn / networkx
│   └── run_all.py            regenerates every figure and table
├── tests/                    232 tests (pytest)
├── web/
│   ├── server.py             JSON API + static workbench
│   ├── workbench.html        Search / Tests / Complexity / Results views
│   ├── run_tests.py          writes report/tables/tests.json
│   └── screenshot.py         UI verification (light, dark, phone)
├── docs/
│   ├── complexity.md         complexity proofs for every component
│   ├── design_decisions.md   every non-obvious choice, what it replaced
│   └── complexity_api_table.md  generated per-function table
└── report/
    ├── report_sections/      section_1.md … section_9.md
    ├── defense_script.md     5-minute defense, measured numbers only
    ├── figures/              PNGs from the benchmarks
    ├── tables/               CSV + Markdown tables, summary.json, transcripts
    ├── diagrams/             Mermaid sources, SVG + PNG diagrams, UI screenshots
    ├── bm25_worked_example.py   regenerates the hand-traced example
    ├── build_diagrams.py        writes the Mermaid and SVG diagrams
    └── generate_report.py       writes the per-function API table
```

---

## Headline measured results

Full detail in `report/report_sections/section_7.md`; every number is in
`report/tables/summary.json`.

| finding | number |
|---|---|
| BM25 vs naive full scan, 18,719 docs | **3,215×** faster (1.446 ms vs 4,648.113 ms); peak **3,215×** at 18,719 docs |
| BM25 agreement with `rank_bm25`, same tokenisation | **99.6 %** top-10 overlap, at 8.8× its query speed |
| Hash table vs `dict` | 136.4× slower insert, 13.7× slower lookup, 1.80× memory |
| Top-K heap vs `heapq` | 4.1× slower, **identical output** |
| K-Means vs sklearn (4,000 docs, k=20) | 1.18× faster, inertia 3,789.9 vs 3,780.1, NMI 0.3458 vs 0.3539 |
| K-Means quality on the full corpus | purity 0.3141, NMI 0.3078, ARI 0.1081 |
| ML pruning speedup (probe 1/2/3) | 1.48× / 1.34× / 1.23× |
| ML pruning Recall@10 (probe 1/2/3) | **29.6 % / 46.9 % / 55.0 %** — the cost, stated |
| BFS vs `networkx` (same 1,759 nodes) | 11.8× slower (12.038 ms vs 1.023 ms) |
| Citation graph | 3,890 cases, 3,077 edges, 17 classes, 1,526 isolated |
| Build | 193.00 s (K-Means 93.74 s) |
| Tests | 232 passing |
| Hash distribution | 21,912 final-table collisions vs 20,917 predicted (1.048) |

---

## Known limitations

* **Lexical only.** No embeddings: `goaltender` does not match `goalie` unless
  the light stemmer maps it. The light Porter stemmer covers steps 1a/1b and a
  trimmed step 4, not the full algorithm.
* **Pruning loses recall.** 29.6–55.0 % of the baseline's top-10 is recovered
  for a 1.23–1.48× speedup. No probe setting is both fast and recall-safe.
* **Weak clusters.** Purity 0.3141 at k=20; K-Means stopped at `max_iter`, not
  at tolerance.
* **Local citation slice.** 22,381 citation records point outside the corpus and
  are dropped; 39 % of cases are isolated.
* **Slow build.** 193.00 s; amortised by the pickle cache, not by the code.
* **Single-user web server.** Loopback-only, `Host`-checked, static whitelist,
  no authentication or concurrency control.
* **Verified on Windows 11 / CPython 3.10 only.** The code is platform-neutral;
  the timings are not portable claims.

---

## Requirements

`requirements.txt` lists only what the *benchmarks and tests* need beyond the
standard library: `numpy`, `matplotlib`, `scikit-learn`, `pytest`, plus
`rank_bm25` and `networkx` for the comparison benchmark. The engine itself uses
nothing but the standard library.
