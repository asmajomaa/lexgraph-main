# 8. Comparison with existing implementations

`benchmarks/compare_existing.py` measures this project against the libraries a
practitioner would reach for. The methodological rule throughout: **the same
input, the same tokenisation, the same k**. The project's `Tokenizer` object is
handed to `rank_bm25` and to sklearn's `TfidfVectorizer`, so a difference in a
table is a difference in implementation and nothing else.

Reproduce with `python -m benchmarks.compare_existing`.

## 8.1 Hash table vs built-in `dict`

150,000 string keys, inserted one at a time, then looked up one at a time;
memory measured with `tracemalloc` around construction.

| measurement | mine | `dict` | ratio |
|---|---|---|---|
| insert 150,000 keys | 1,850.65 ms | 13.56 ms | **136.4× slower** |
| lookup 150,000 keys | 433.01 ms | 31.51 ms | **13.7× slower** |
| peak memory | 13,825 KiB | 7,681 KiB | **1.80× more** |

`dict`'s build time is measured as a separate `dict` comprehension rather than
inside the same loop, so the comparison is insert-versus-insert; an earlier
version of this benchmark measured the dict benchmark's *total* wall time
including key generation and printed a negative memory delta, which is why the
harness now separates construction from measurement.

**Where the 120× goes.** CPython's `dict` is C code with open addressing and a
compact key/value array; mine is a Python loop over `_Entry` objects with a
polynomial hash per key. The collision counts show the hash function is *not*
the problem: on this benchmark's 150,000 synthetic keys the finished table has
35,863 chain collisions against 35,779 predicted by the uniform model (1.002),
and on the real 112,184-word vocabulary 21,912 against 20,917 (1.048). The
constant factor is interpreter overhead. (The same table's *lifetime* counter
reads 94,238 for the benchmark keys because they are inserted one `put` at a
time and every resize re-inserts them; section 5.1 explains the two metrics.)

**Where the 14.4× goes** is the same story, and the lookup ratio is smaller than
the insert ratio because a lookup pays one hash instead of a hash plus an
allocation plus an occasional resize. The memory ratio (1.80×) is the price of
one `_Entry` object per key (with `__slots__`) plus the bucket array, against
`dict`'s compact layout.

**What the custom table buys.** Four measurements `dict` cannot give: a
`collisions` counter (lifetime inserts), a `chain_collisions` counter (final
table shape), a `probes` counter, and a `load_factor` the benchmark can cite.
Section 7.2's claim that the hash distributes well is *measured from the
table's own instrumentation*, not inferred — and the instrumentation is what
revealed that a lifetime counter mixes rehash work into the collision count,
which is why the second metric exists.

## 8.2 Bounded heap vs `heapq`

200,000 random `(score, doc_id)` candidates, `k = 10`, three repeats (`compare_heap(repeats=3)`, the committed default).

| measurement | mine | `heapq` | ratio |
|---|---|---|---|
| time | 66.96 ms | 16.16 ms | **4.1× slower** |
| same top-10? | **yes** (`same_result: True`) | — | — |

The 4.1× is again interpreter overhead; the algorithm is identical in both
(`nlargest` uses a bounded min-heap internally). This is the cleanest
"same answer, different constant" result in the project, and it is worth being
explicit that the *output* equality is checked, not assumed — the test suite
compares against a full-sort oracle on every random stream and every `k`, and
the benchmark compares against `heapq.nlargest` on this one.

## 8.3 BM25 vs `rank_bm25` and sklearn

100 queries sampled from document titles, `k = 10`, identical tokenisation
passed into both references.

| implementation | build / fit | query time | ratio vs mine | top-10 agreement |
|---|---|---|---|---|
| **custom BM25** | index built at engine build | **1.346 ms** | 1.00 | reference |
| `rank_bm25.BM25Okapi` | 534.35 ms | 11.813 ms | **0.11×** (mine is 8.8× faster) | **99.6 % overlap** |
| sklearn `TfidfVectorizer` + cosine | 4,582.34 ms | 27.810 ms | **0.05×** (mine is 20.7× faster) | **65.2 % overlap** |

Two results with two different explanations:

* **The 99.6 % agreement with `rank_bm25` is the important number.** It says my
  BM25 is the same scoring function: same idf, same saturation, same length
  normalisation. The 0.4 % disagreement is the tie-breaking rule (I break ties
  by lower `doc_id`, `rank_bm25` by its sort order) and the rare term whose
  stemmed form differs, not a different formula.
* **The speed advantage is real but must be read carefully.** Both references
  score *dense*: `rank_bm25.get_scores` walks every document for every query
  (it is a scoring library, not an inverted-index retrieval engine), and the
  sklearn path calls `cosine_similarity` against the whole 18,719 × V matrix.
  My scorer walks only the postings of the query's terms. So the comparison is
  "inverted index versus dense scorer" as much as "my code versus their code" —
  the 8.8× is an architectural difference, not a claim that Python beats a
  library. Against a C implementation *with* an inverted index the result
  would reverse, and section 8.6 says so.
* **The 65.2 % overlap with sklearn is not a bug.** sklearn computes cosine
  similarity over TF-IDF, which weights terms differently from BM25 (no
  saturation, no length normalisation in the same form) and does not
  implicitly satisfy the query. Two *different* ranking functions agree on
  roughly two thirds of a top-10; that is the expected overlap between BM25 and
  cosine TF-IDF on short documents, and it is why both are legitimate baselines.

## 8.4 K-Means vs `sklearn.cluster.KMeans`

4,000-document sample of the newsgroups corpus, `k = 20`, `random_state = 42`,
both fixed to the same maximum of 60 iterations.

| implementation | fit time | inertia | NMI vs labels |
|---|---|---|---|
| **custom K-Means** | **10,725 ms** | 3,789.9 | 0.3458 |
| `sklearn.cluster.KMeans` | 12,696 ms | **3,780.1** | **0.3539** |

* **Time: mine is 1.18× faster**, which surprised me and needs stating plainly.
  sklearn's implementation is C-backed but does dense arithmetic on 4,000 ×
  5,000 floats; mine uses the sparse-dense identity and therefore touches only
  the non-zeros, and on TF-IDF vectors that dominates the constant-factor
  advantage. This result is *sample-size dependent* and should not be
  generalised: at full corpus size my implementation takes 93.74 s (section
  7.1), and a dense C implementation would almost certainly win there.
* **Quality: sklearn is slightly better** — inertia 3,780.1 against 3,789.9
  (0.26 % lower) and NMI 0.3539 against 0.3458. That is the expected shape:
  sklearn runs `n_init = 10` restarts and keeps the best, mine is one seeded
  run. **I switched the comparison to `n_init = 10` deliberately** — an earlier
  version used `n_init = 1` and my implementation "won" the NMI comparison,
  which was flattering nonsense caused by giving the reference one restart.
  The committed comparison gives sklearn its default and reports that it wins
  on quality.
* **The NMI values differ from section 7.6's 0.3078** because this is a 4,000-
  document sample, not the full corpus. Both are real; the sample number is
  used only for the library comparison.

## 8.5 BFS vs `networkx`

Traversal from `06_584` on the real 3,890-node / 3,077-edge graph, both
implementations reaching the same **1,759 nodes**.

| implementation | per traversal | first call (includes index) | ratio |
|---|---|---|---|
| **custom BFS** | 12.038 ms ± 0.278 | 26.49 ms | 1.00 |
| `networkx.bfs_edges` | **1.023 ms ± 0.353** | — | **11.8× faster** |

The first-call column is separate on purpose. The very first traversal also
assigns the graph's node index — one polynomial hash per node over 3,890
string ids, then a dense integer key for the bytearray. An earlier version of
this benchmark timed that setup *inside* the repeat loop, which averaged a
one-off O(V) cost into a per-traversal figure and produced "cold" numbers
anywhere from 15 ms to 76 ms on the same machine. Separating them makes both
numbers stable: 12.04 ms to traverse, 26.5 ms
extra for the one-time index a freshly loaded engine pays once.

`networkx` wins clearly, as expected: its adjacency is a C-level dict of dicts
and the traversal is a tight Python loop over C structures, against my
Python-object chain walk plus a bytearray. The 11.8× is the price of the
hand-built adjacency, and the honest conclusion is the brief's: **mine is
slower, but every step of it is visible and changeable.**

There is a caveat worth recording: `networkx` was measured on the same
*logical* graph but not the same *object*, so a fair reading is "the traversal
algorithms differ by this much on this graph", not "my adjacency is 11.8×
worse" — the adjacency construction is not included in either timing.

## 8.6 The honest conclusion

![Every comparison at once, on a log scale](../figures/fig_5_comparison.png)

The figure is the section in one picture: seven bars, each "my time ÷ the
reference's time" on a multiplicative axis, with the dashed line at 1.0. Bars to
the right of the line are slower than the library; bars to the left are faster.
Reading it left to right, the same three-way story the table tells below is
visible at a glance — three losses to C, two wins on algorithms that differ, one
near-parity. A linear axis was rejected because the 136× hash result would
compress every other bar beyond legibility.

| component | vs the reference | verdict |
|---|---|---|
| hash table | 136.4× slower insert, 13.7× slower lookup, 1.80× memory | **slower — expected**: pure Python objects vs C open addressing |
| heap | 4.1× slower, identical output | **slower — expected**: same algorithm, interpreter overhead |
| BM25 | 8.8× faster than `rank_bm25`, 20.7× faster than sklearn TF-IDF, 99.6 % agreement | **faster, and equivalent where it matters** |
| K-Means | 1.18× faster on a 4,000-doc sample, 0.26 % worse inertia, 0.0081 worse NMI | **competitive on time, slightly worse on quality** (one seed vs ten restarts) |
| BFS | 11.8× slower | **slower — expected**: Python adjacency vs C dicts |

So the brief's predicted summary — *"mine is slower (pure Python vs C), but
transparent and customisable"* — holds for the hash table, the heap and BFS,
and does **not** hold for BM25 or K-Means, where the hand-written version is
faster on this workload. The report states both directions rather than
manufacturing a uniform loss. The two places where being slower is a real
problem (hash insert during build, BFS on large graphs) are visible in the
build timings: the newsgroups index takes 8.57 s and the legal index 48.54 s,
which is acceptable at this scale and would not be at 10× this scale.

The transparency claim is not rhetorical. Concretely, the custom structures
made possible measurements no library offers:

* collision and probe counters that let section 4.4.1 check the `1 + α` bound
  against real data;
* a `scored_documents` counter on `BM25`, which is the only reason the ML path
  can report "413 of 18,719 documents scored" instead of estimating;
* an `AgentResult` carrying per-step timings (`vectorize_ms`, `probe_ms`,
  `score_ms`, `bfs_ms`), which is what let me find and fix the harness bug that
  had been timing the baseline inside the ML path;
* a `ClusterFilter` that could be killed and rewritten three times against
  measured numbers rather than a hunch.
