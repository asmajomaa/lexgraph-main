# 6. Implementation of the algorithm

This section walks the runtime code, not the pseudocode: what `main.py` calls,
what the agent dispatches to, and how a query becomes a ranked list. Code is
quoted verbatim.

## 6.1 The entry point

```bash
python main.py --query "hockey playoffs goalie" --mode ml --k 5 --compare
```

`main.py` parses arguments, calls `get_engine()` (which loads the pickle if its
fingerprint matches `config.py`, and otherwise rebuilds), then

```python
result = engine.query(
    query, mode=args.mode, k=args.k,
    probe_clusters=args.probe_clusters, bfs_depth=args.bfs_depth,
    compare_baseline=args.compare,
)
```

`Engine.query` delegates to `self.agent.run(...)` — the agent is the only object
that knows the three paths exist.

## 6.2 The ReAct loop

`core/agent.py:603`:

```python
started = time.perf_counter()
intent, reason_logs = self.reason(query, mode)
act_started = time.perf_counter()
path, payload = self.act(query, intent, mode, k, probe_clusters, bfs_depth)
act_ms = (time.perf_counter() - act_started) * 1000.0

baseline_ranked: Optional[List[Tuple[float, int]]] = None
baseline_ms = 0.0
if compare_baseline and path == "ml":
    baseline_started = time.perf_counter()
    baseline_ranked = self.news_scorer.search(query, k)
    baseline_ms = (time.perf_counter() - baseline_started) * 1000.0

rows, observe_logs = self.observe(path, payload, k, baseline_ranked)
text = self.answer(path, rows, query)
```

### Reason — a word-level rule, deliberately not a substring test

```python
def is_relational(query: str) -> bool:
    lowered = query.lower()
    words = set()
    current: List[str] = []
    for char in lowered:
        if "a" <= char <= "z":
            current.append(char)
        else:
            if current:
                words.add("".join(current))
                current = []
    if current:
        words.add("".join(current))
    for keyword in RELATIONAL_KEYWORDS:
        if keyword in words:
            return True
    return False
```

The tokenisation to `words` is not decoration. A substring test
(`"cite" in query`) fires on **"excited"** and on **"precedence"**; both are
real lexical queries users type, and both would be routed to the citation graph
where they return nothing. Matching whole words only costs one pass over the
query and removes the failure entirely. `tests/test_agent.py` pins the
behaviour with those two words as explicit negative cases.

The keyword list lives in `config.RELATIONAL_KEYWORDS` (14 entries: `cite`,
`cites`, `cited`, `citing`, `citation`, `citations`, `reference`, `references`,
`referenced`, `followed`, `applied`, `distinguished`, `precedent`,
`precedents`). It is a rule, not a classifier: it cannot be wrong about a
query it did not see, and it is trivially auditable — two properties an
LLM-based router would not have, at the cost of not understanding paraphrase
("papers building on X" routes to lexical). Section 9 treats that as a
limitation, not a feature.

### Act — dispatch

```python
if intent == "RELATIONAL":
    return "relational", self._act_relational(query, k, bfs_depth)
if mode == "ml" and self.kmeans is not None and self.vectorizer is not None:
    return "ml", self._act_ml(query, k, probe_clusters)
return "baseline", self._act_baseline(query, k)
```

Three lines, three paths, one ordering rule that matters: **intent beats
mode**. Asking "which cases cite 06_584" with `--mode ml` still takes the graph
path, because the ML path prunes *lexical* candidates and cannot answer a
relational question. The PATH line keeps both facts visible:

```text
[AGENT][REASON] intent=RELATIONAL: relational keyword(s) ['cite'] -> citation-graph path
PATH    intent=RELATIONAL  path=relational  mode=ml  total=27.57 ms
```

i.e. the request asked for `ml`, the agent chose `relational`, and the extra
`mode=ml` in the line makes that divergence auditable rather than silent.

### Baseline path

```python
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
    f"({self._pct(...)}) in {elapsed:.2f} ms",
]
```

`candidates_scored` is read from the scorer, not estimated: `BM25.search`
increments `scored` inside the posting loop only when a term actually
contributes, so "scored 475 documents (2.5%)" in the transcript means 475
documents accumulated a non-zero score — the other 18,244 contributed nothing
to this query. That distinction matters for section 7: the baseline does not
"score the whole corpus", it *visits* the whole postings of its terms and
scores those 475.

### The scorer underneath — `core/bm25.py`

Every path above ends in `BM25.search`, so the scoring code itself belongs here.
First the two scoring primitives, `core/bm25.py:72` and `core/bm25.py:98`
(docstrings elided for length; the bodies are otherwise verbatim):

```python
def idf(self, document_frequency: int, n_docs: int) -> float:
    if document_frequency <= 0 or n_docs <= 0:
        return 0.0
    ratio = 1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5)
    if ratio <= 1.0:
        return 0.0
    return math.log(ratio)

def term_score(
    self,
    term_frequency: int,
    document_length: int,
    average_length: float,
    idf: float,
) -> float:
    if term_frequency <= 0 or idf == 0.0:
        return 0.0
    if average_length <= 0.0:
        average_length = 1.0
    numerator = term_frequency * (self.k1 + 1.0)
    denominator = term_frequency + self.k1 * (
        1.0 - self.b + self.b * document_length / average_length
    )
    return idf * numerator / denominator
```

`term_score` is the whole saturation argument in five lines: as `tf` grows the
numerator and denominator both grow, so the contribution approaches
`idf · (k1 + 1)` — a document cannot win by repeating a term. `dl / avgdl` is the
only place `length` enters, and with `b = 0.75` and `dl` far above `avgdl` the
denominator grows, which is the penalty.

The retrieval loop, `core/bm25.py:181` to `core/bm25.py:230`, verbatim and
dedented one level (the whole body sits inside `def search`):

```python
terms, idf_by_term = self._prepare_query(query)
n_docs = self.index.N
if not terms or n_docs == 0:
    self.scored_documents = 0
    self.last_stats = {"terms": 0.0, "scored": 0.0, "postings_visited": 0.0}
    return []

scores = [0.0] * n_docs
doc_lengths: List[int] = self.index.doc_lengths  # type: ignore[attr-defined]
average_length: float = self.index.average_document_length  # type: ignore[attr-defined]
candidates = getattr(self.index, "candidates", None)
pool = getattr(self.index, "pool", None)
is_candidate = candidates.contains if candidates is not None else None
postings_visited = 0
scored = 0

for term in terms:
    idf = idf_by_term[term]
    for doc_id, tf in self.index.postings_for(term):
        postings_visited += 1
        if is_candidate is not None and not is_candidate(doc_id):
            continue
        contribution = self.term_score(tf, doc_lengths[doc_id], average_length, idf)
        if contribution:
            scores[doc_id] += contribution
            scored += 1
            if pool is not None:
                pool.mark(doc_id)

# When scoring against a pool bitmap, iteration is bounded by the
# candidates actually touched rather than the whole corpus.
heap = TopKHeap(k)
if pool is not None:
    for doc_id in pool.ids():
        score = scores[doc_id]
        if score > 0.0:
            heap.push(score, doc_id)
else:
    for doc_id in range(n_docs):
        score = scores[doc_id]
        if score > 0.0:
            heap.push(score, doc_id)

self.scored_documents = scored
self.last_stats = {
    "terms": float(len(terms)),
    "scored": float(scored),
    "postings_visited": float(postings_visited),
}
return heap.get_top_k()
```

Three decisions are visible in that loop and each was made against a measurement.

* **Term-at-a-time, not document-at-a-time.** `scores` is a dense `list` indexed
  by `doc_id`, so a posting visit is one index and one add; the document-at-a-time
  alternative (`score_document`, kept as an exact single-document oracle for the
  tests, and `explain`, which backs `--explain`) would re-walk every posting
  list per document. Section 8.3 is the payoff: 1.351 ms per query against
  `rank_bm25`'s 11.973 ms.
* **The candidate test is skipped before the arithmetic, not after.** On the ML
  path `is_candidate` is the `ClusterFilter` mask, so a pruned document costs one
  comparison and never reaches `term_score`. That ordering is what makes
  "413 of 18,719 scored" true rather than rhetorical.
* **The heap is filled from the score array, not inside the posting loop.** Pushing
  from inside would push the same document repeatedly, once per query term; the
  two-pass form pushes each scorer once, which is why the heap comparison is
  `O(D log K)` with `D` the corpus, and `O(|pool| log K)` under pruning.

`postings_visited` and `scored` are the counters that make the ML path's
`[AGENT][OBSERVE]` line a measurement rather than an estimate: the transcript
distinguishes "walked 1,082 postings" from "scored 165 documents", and the gap
between those two numbers is exactly the recall cost the pruning pays.

### ML path

```python
query_vector = self.vectorizer.transform_text(query)
vectorize_ms = (time.perf_counter() - started) * 1000.0

nearest = self.kmeans.predict_top_n(query_vector, max(1, probe_clusters))
clusters = [cluster for cluster, _distance in nearest]
candidates = ClusterFilter(self.clusters, clusters, n_clusters=self.kmeans.k)

view = self.news_index.restrict(candidates)
scorer = BM25(view)
ranked = scorer.search(query, k)
```

The order is the whole point: the query is scored against 20 centroids
(`O(k·nnz)`), the two nearest clusters become a 20-byte mask, and BM25 then
walks the same postings as the baseline but *skips* every document outside the
mask before doing any arithmetic.

One incident belongs here because it is the honest version of the ML-path
story. My first version built a `CandidateSet` of the whole probed pool, which
cost ~0.9 ms per query (1.668 ms when re-measured for section 5.9 over the two
nearest clusters of a real query) — more than the search it was accelerating —
and the measured "speedup" was 0.54×, i.e. *slower than baseline*.
Investigating that produced three separate fixes (the cluster mask, the
global-df view, and excluding the comparison baseline from the timed region of
`run`), after which the honest number is 1.48× at probe 1, with the recall
loss stated in the same breath. An early benchmark that reported 17× was wrong
for a completely different reason: it timed the baseline *inside* the ML path's
own measurement window. Neither error survives in the committed numbers, and
section 9 keeps the lesson.

### Relational path

```python
seed_id = self._extract_case_id(query)
...
if seed_id is None:
    resolution = "bm25_top1"
    fallback = self.legal_scorer.search(query, 1)
...
traversal = self.graph.bfs_traverse(
    seed_external, max_depth=bfs_depth, edge_type=None, direction="both"
)
```

Two resolution modes, both logged:

* **`resolution=query`** — a case id (`06_584`), a neutral citation
  (`[2006] FCA 584`), or a distinctive fragment of a case name appears in the
  query. `LegalCorpus.resolve_case` tries an exact alias, then the citation
  key, then the stripped form, then a unique substring.
* **`resolution=bm25_top1`** — no id found, so the query is run as a BM25
  search against the *legal* index (not the newsgroups one) and the top case
  becomes the seed. The log says so, because a user reading the trace must be
  able to tell whether the seed was chosen or inferred.

`direction="both"` is the substantive choice. Australian citation practice is
asymmetric: `06_584` has in-degree 99 and out-degree 0, so a forward-only
traversal from it would return the seed and nothing else. Asking in both
directions is what makes "which cases cite X" and "what does X rely on"
answerable with one traversal — and the BFS deduplicates, so a case that both
cites and is cited by the seed appears once, at its minimum hop distance.

### Observe

```python
logs: List[str] = []
if path == "relational":
    rows = list(payload.get("nodes", []))
    logs.append(
        f"[AGENT][OBSERVE] returning {len(rows)} connected case(s) "
        f"including the seed"
    )
    return rows, logs

ranked: List[Tuple[float, int]] = payload.get("ranked", [])
index = self.news_index
rows = []
for score, doc_id in ranked[:k]:
    rows.append({
        "doc_id": doc_id,
        "external_id": index.external_ids[doc_id],
        "title": index.titles[doc_id],
        "snippet": index.snippets[doc_id],
        "score": score,
        "label": index.labels[doc_id],
    })
if baseline_ranked is not None and path == "ml":
    baseline_ids = [doc_id for _score, doc_id in baseline_ranked]
    overlap = len({row["doc_id"] for row in rows} & set(baseline_ids))
    logs.append(
        f"[AGENT][OBSERVE] recall vs baseline: {overlap}/{len(baseline_ids)} "
        f"baseline results recovered ({100.0 * overlap / max(1, len(baseline_ids)):.1f}%)"
    )
```

`observe` is where "what did the path actually do" becomes a number the user can
read. The recall line only appears when a baseline was run, and it is computed
as a *set* overlap on doc ids, not on scores — pruning can move ranks within
the returned set, and counting rank changes would conflate two different
failures.

### Answer

```python
if path == "relational":
    seed = rows[0]
    others = [row for row in rows if row.get("depth") not in (0, None)]
    seed_external = str(seed.get("external_id"))
    citing = [row for row in others if str(row.get("parent")) == seed_external]
    edge_classes = sorted(
        {str(row.get("edge_type")) for row in others if row.get("edge_type")}
    )
    classes_text = f" (edge types: {', '.join(edge_classes)})" if edge_classes else ""
    return (
        f"Starting from {seed.get('title')!r} ({seed.get('external_id')}), a breadth-first "
        f"walk of the citation graph reached {len(others)} connected case(s) within the "
        f"depth limit{classes_text}; {len(citing)} of them are directly connected to the "
        f"seed (depth 1). Results are ordered by hop distance."
    )
top = rows[0]
score = float(top.get("score") or 0.0)
return (
    f"{len(rows)} document(s) matched {query!r}. The strongest is "
    f"{top.get('title')!r} (BM25 {score:.3f}) from the "
    f"{top.get('label') or 'newsgroups'} category."
)
```

The relational answer counts `others` (everything at depth ≥ 1), not the direct
children. An earlier version said "reached 0 connected case(s)" while listing
nine results, because it counted only `rows[1:]` whose parent was the seed; on
this graph the seed's neighbours are reached via the *reverse* edge, so their
`parent` is the seed but their direction differs. The fix was to separate the
two counts — total reachable vs directly connected — and name the edge classes,
which is more informative anyway.

## 6.3 The exact code path for the two transcript queries

**`hockey playoffs goalie`, mode `ml`, probe 2** (section 7 transcript):

1. `reason` tokenises to `{hockey, playoffs, goalie}`, finds no relational
   keyword → `LEXICAL`.
2. `act` sees `mode == "ml"` → `_act_ml`.
3. `vectorizer.transform_text` → `nnz = 3` (all three terms are in the 5,000-term
   vocabulary), ~0.05 ms.
4. `kmeans.predict_top_n(v, 2)` → clusters `[2, 1]`, `O(20 · 3)` dots.
5. `ClusterFilter` masks clusters 1 and 2; the pool is 5,597 documents (29.9 %).
6. `BM25.search` walks the three posting lists, skips 5,597 minus the 413 that
   score, pushes 413 candidates through `TopKHeap(5)`.
7. `observe` reports recall 5/5 against the baseline; `answer` names
   `Where's Roger?` at 23.113.

**`which cases cite 06_584`** (section 7 transcript):

1. `reason` finds `cite` as a whole word → `RELATIONAL`.
2. `act` ignores the mode and calls `_act_relational`.
3. `_extract_case_id` finds `06_584` → `resolution=query`.
4. `bfs_traverse("06_584", max_depth=2, direction="both")` → 194 nodes,
   16.67 ms; the graph has 3,890 nodes / 3,077 edges.
5. `observe` returns them ordered by depth; `answer` reports 4 others at
   depth 1 with edge type `cited`.

## 6.4 Configuration as a first-class object

Every constant used above is imported from `config.py`, and each carries a
comment stating why that value: `BM25_K1 = 1.5` ("middle of Robertsson &
Zaragoza's recommended band"), `HASH_LOAD_FACTOR_MAX = 0.75` ("keeps average
chain length near 1.5 probes while leaving enough head-room that resizes are
rare"), `DEFAULT_PROBE_CLUSTERS = 2` ("probing 1 is fastest but loses recall;
probing all k degenerates to the baseline"), `KMEANS_MAX_ITER = 60`. The
benchmark loops read `BENCH_REPEATS` and `EVAL_QUERY_COUNT` from the same
module, so the numbers in section 7 are reproducible by editing one file.

## 6.5 The workbench as a second front end

`web/server.py` runs a `ThreadingHTTPServer` exposing the same agent over JSON:

| endpoint | method | purpose |
|---|---|---|
| `/api/info` | GET | corpus, index and cluster metadata |
| `/api/query` | POST | `{query, k, probe_clusters, mode}` → the full `AgentResult.to_dict()` |
| `/api/tests` | GET | the last pytest run, read from `report/tables/tests.json` |
| `/api/complexity` | GET | the operation table plus the measured series from `summary.json` |
| `/api/figures` | GET | the figure/table inventory under `report/` |
| `/api/build` | POST | rebuild the engine in a background thread |

Two security details, because the server binds a loopback port but is still a
network listener:

* **DNS-rebinding defence.** Every request whose `Host` header is not
  `127.0.0.1`, `localhost`, `::1` (or absent) is answered with `403` — a page
  on a hostile domain cannot make the browser resolve an attacker domain to
  127.0.0.1 and read this API.
* **Static whitelist.** File serving is restricted to
  `{.png, .svg, .csv, .md, .json, .css, .js}` under a fixed set of roots, so a
  crafted path cannot reach `cache/engine.pkl` or a source file.

The screenshot harness (`web/screenshot.py`) fails the run if the page raises a
JavaScript error or scrolls horizontally at 400 px width; both are reported in
section 7.
