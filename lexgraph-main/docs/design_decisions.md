# Design decisions

Every non-obvious choice, what it replaced, and the evidence that made it stick.
Numbers come from the committed benchmark run (`report/tables/summary.json`) or
from measurements recorded in the report sections named.

---

## 1. Simulation and process

### 1.1 Build the structures from scratch, in pure Python

**Decision.** No `dict` or `set` used as a lookup structure anywhere in
`structures/`, `core/` or `ml/`; no `heapq`, `networkx`, `scikit-learn` or
`numpy` either. The two lookup-table alternatives it allows instead are the
project's own `HashSetTable` and, where a list is genuinely the better
structure, a sorted tuple with `bisect`.

**Why.** The brief grades the structures themselves, and a library hides the
three things that make a data-structure project worth reading: the collision
counts, the probe counts, and the choice of load factor. Those counters are not
decoration — section 4.4.1 checks the `1 + α` bound against 112,184 real keys
because the table counts its own collisions.

**Cost.** Enforced on three fronts by `tests/test_purity.py`: an import
audit, a *construction* audit that rejects empty `{}`/`set()`, comprehensions
and zero-argument `dict()`/`set()` inside engine functions, and behavioural
assertions that each index-like table is the custom class. `benchmarks/` and
`tests/` are exempt by design — they exist to check the core against those very
libraries.

**What the stricter rule cost, measured.** Three real changes, not just a
tightened test. The vectoriser's `term -> column` map became a `HashSetTable`.
The tokenizer's stop-word membership became a sorted tuple + `bisect`, which is
*faster* than the hash table on 150 words (0.254 µs against 0.676 µs per probe)
because the table pays a Python frame and a polynomial hash per token. And
`term_frequencies` dropped its lookup table entirely for sort-and-count, since a
`dict` costs 2.3 ms, `HashSetTable` 10.9 ms and the sort 1.9 ms on a
3,445-token legal document. The full build stayed at 193.00 s and K-Means
converged to the identical inertia of 17888.09, so no result changed.

### 1.2 `config.py` holds every tunable

**Why.** "No unexplained magic numbers" is a grading criterion, and a constant
inside an algorithm is an undocumented decision. `BM25_K1 = 1.5` carries the
published range in a comment; `HASH_LOAD_FACTOR_MAX = 0.75` carries the probe
argument; `KMEANS_MAX_ITER = 60` is referenced by both the engine and the
benchmarks, so editing one file changes both consistently.

### 1.3 Pickle for the cache, with a parameter fingerprint

**Decision.** One `pickle` file (`cache/engine.pkl`, 219.4 MB) of plain lists,
tuples, strings and numbers, written through each structure's `to_state()`.

**Alternatives.** A line-oriented text format (3× the size, a parser with its
own bug surface, no portability benefit for a machine-local cache); one file
per structure (more moving parts, no atomic replace).

**Invalidation.** A fingerprint of the build parameters is stored beside the
payload and compared with `config.py` on load, so changing `KMEANS_K` or the
vocabulary size rebuilds instead of silently serving a stale model. The
on-disk *schema* is versioned separately (`CACHE_VERSION`).

**Risk accepted.** Pickle is not portable across Python versions or a changed
class layout; both are documented in the README, and the fix is `--build`.

---

## 2. Hash table

### 2.1 Polynomial rolling hash, base 31, mod 2⁶¹−1

**Why 31** — prime; separates anagrams (`"ab"` → 3105, `"ba"` → 3135);
`31·h` fits a machine word for realistic tokens, so CPython stays in its
one-digit integer path and each character costs one multiply-add.

**Why 2⁶¹−1** — a Mersenne prime: the polynomial is evaluated over a field, and
`h·31 + ord(c) < 2⁶³` for any realistic key, so no big-int arithmetic appears
in the hot loop. `2⁶³` is not prime; `2³²` overflows int64 sooner.

**Evidence the hash is good** — 112,184 keys in 262,144 buckets, **21,912
final-table collisions** against 20,917 predicted by the uniform model (ratio
1.048), mean chain 0.428, longest chain 6. The hash is not why the table loses
to `dict`. (The lifetime counter reads 83,222 for the same keys because resizes
re-insert them; see section 5.1 of the report for why both numbers exist.)

### 2.2 Separate chaining, not open addressing

**Why** — deletion is a pointer splice rather than a tombstone (a tombstone
lengthens every later probe and breaks the `1 + α` bound); degradation under a
clumpy hash is linear and predictable rather than a probe cliff; and the load
factor can sit at 0.75 without the probe explosion open addressing suffers
there.

**Cost** — a `_Entry` object per key: peak memory 13,825 KiB against `dict`'s
7,681 KiB (1.80×), and insert 136.4× slower (section 8.1).

### 2.3 Load factor 0.75, capacity a power of two, no shrinking

* **0.75** keeps expected probes at `1 + α/2 = 1.375` while resizes stay rare;
  it also guarantees `capacity ≤ (4/3)n`, which is what makes the space `O(n)`.
* **Power of two** lets `hash & (capacity − 1)` replace a second modulo; the
  polynomial hash already reduced modulo a prime, so the low bits are mixed.
* **No shrinking** avoids capacity oscillation around the threshold (repeated
  rehashes from an alternating insert/delete pattern) at the cost of memory that
  is never returned — documented in `delete`.

### 2.4 `PostingsTable` stores Python lists per term

**Decision.** `term -> [(doc_id, tf), ...]` as a plain list appended during
build and iterated read-only at query time.

**Why** — a hand-rolled linked postings list would give the same asymptotics
with worse locality and no benefit: build is append-only and query is sequential
iteration. Postings are in ascending `doc_id` because documents are added in
id order, which is what would allow a merge-style intersection later.

---

## 3. Heap

### 3.1 Bounded min-heap of the best-so-far

**Why** — to keep the *top* K, hold the *worst* survivor at the root: one
comparison rejects almost every candidate. A max-heap would put the best item at
the root and require a pop to find the worst.

**Cost/benefit** — `O(N log K)` time and `O(K)` space against `O(N log N)` and
`O(N)` for a full sort. On 200,000 candidates with `K = 10` it is 66.41 ms
against `heapq`'s 16.03 ms with **identical output**.

### 3.2 Deterministic tie-break by lower `doc_id`

**Why** — equal BM25 scores are common on short documents; without a total
order the result would depend on posting order and the report's numbers would
not be reproducible. Lower `doc_id` also makes the `heapq` cross-check
meaningful — but note `heapq` orders tuples lexicographically, so the reference
in the test keys on `(score, 0)` and compares scores only.

---

## 4. Tokenizer

### 4.1 Light Porter (steps 1a, 1b, trimmed 4), not a suffix table

**History.** The first version used an ad-hoc suffix list and produced
`running → rune`, `rendering → rendere`. It was replaced with the actual Porter
steps and the measure rule. `run`, `render`, `write`, `study`, `policy`,
`class`, `news` are all unchanged and pinned by tests; `applied → apply`.

**Why not the full algorithm** — steps 2, 3 and 5 add ~200 lines for a small
measured gain on this corpus. The cost is that derivational pairs like
`generalisation`/`generalise` do not collapse (section 9.3).

### 4.2 An embedded stop-word tuple, not a downloaded list

**Why** — a fixed 150-word tuple is auditable, has no network dependency, and
keeps the engine's import surface to the standard library. The duplicate `don`
entry found during review was removed.

### 4.3 Character-at-a-time scanning, apostrophes dropped

**Why** — `re` tokenisation would be shorter but the brief asks for a
hand-written tokenizer and a regex would have to be re-explained in the report.
Dropping apostrophes makes `don't → dont` (which then matches "dont" posts)
rather than splitting it into `don` + `t`, where `t` is dropped by min length
and `don` is a stop word.

---

## 5. BM25

### 5.1 Lucene's `+1` idf

**Decision.** `idf = ln(1 + (N − n + 0.5)/(n + 0.5))`.

**Why** — the classical form is negative for terms in more than half the
corpus, which inverts their contribution. The shifted form keeps `idf ≥ 0`, so a
corpus-wide term contributes exactly 0 instead of a negative score. Side effect:
`idf(N, N) = 0`, which contradicted one of my own tests and is now documented.

### 5.2 Term-at-a-time with a dense score array

**Why** — one pass per posting, one array index per accumulation, one
contiguous sweep to feed the heap. The `O(N)` allocation per query is the price,
and it is stated as a limitation of the pruning path (which cannot avoid it
either).

### 5.3 Document-at-a-time for `explain`

**Why** — the report needs hand-checkable per-term numbers, and a
document-at-a-time scorer is trivially correct for one document. A test asserts
that the two strategies agree on random corpora, so the trace in section 3.3
describes the same function the search uses.

### 5.4 `RestrictedView` returns the **global** document frequency

**Decision.** The pruned index reports the same `df` and `idf` as the full one.

**Why** — two reasons. (a) Candidate-local `df` would require walking every
posting of every query term per query, erasing the pruning saving. (b) Using the
same idf on both paths is what makes Recall@10 meaningful: a different idf
would change the ranking for reasons unrelated to pruning, and the comparison
would measure two changes at once.

---

## 6. Vectoriser

### 6.1 Top-5,000 terms by document frequency, `min_df = 2`, `max_df = 0.5`

**Why** — 5,000 keeps the centroid matrix at 800 kB (20 × 5,000 × 8 B) and the
sparse-dense distance identity effective; `min_df = 2` drops hapax forms that
only add noise and dimension; `max_df = 0.5` is a second stop-word filter for
tokens the tuple missed.

**Determinism** — terms are ranked by `(-df, term)`, so a tie is broken
alphabetically and the vocabulary is identical across runs.

### 6.2 Smoothed idf, log-tf damping, L2 normalisation

`idf = ln((1 + N)/(1 + df)) + 1` keeps every term positive and never divides by
zero; `log(1 + tf)` damps repetition without the magic exponent of `lnc.ltc`;
L2 normalisation makes Euclidean K-Means on the unit sphere act as spherical
K-Means, i.e. cosine ordering — the same metric BM25 approximates.

---

## 7. K-Means

### 7.1 k-means++, `n_trials = 2`

**Why** — uniform seeding drops several centres into the same dense region on
this corpus and Lloyd cannot recover. D² sampling puts the next centre where
coverage is worst; two trial draws per step give the greedy `O(log k)`
guarantee at negligible cost against 60 iterations.

### 7.2 Sparse-dense distance identity

`‖x − c‖² = ‖x‖² + ‖c‖² − 2x·c`, with `‖x‖²` cached on the vector and `‖c‖²`
recomputed once per iteration. Cost drops from `O(k·M)` to `O(k·nnz)` per
document; with `M = 5,000` and `nnz ≈ 60` that is the difference between
finishing and not. Measured fit: 93.74 s at full scale.

### 7.3 Dense centroids

**Why not sparse** — every centroid is compared to every document, so the hot
loop wants contiguous floats. 800 kB for the whole model is affordable; a sparse
centroid representation would save memory and lose on the hot path.

### 7.4 Empty cluster reseeds to the farthest point

**Alternatives.** Re-randomise (lands on a direction no document occupies on
the unit sphere, so it empties again); leave empty (the model silently becomes
`k − 1` clusters while the agent still probes `k`).

**Chosen** — the point farthest from its own centroid, i.e. where the model is
doing worst. `O(n)` with the cached norms.

### 7.5 Two convergence criteria, and an honest `max_iter`

No-label-change (cheap, common case) or max centroid shift below `1e-4` (slow
drift). The committed run hit `max_iter = 60` with inertia still moving, and the
engine reports `converged = "max_iter"` — not "converged".

### 7.6 Fixed seed 42

Every clustering number in this report is reproducible. `n_init = 1` on one
seed is why the sklearn comparison gives sklearn `n_init = 10` (its default):
the stricter test, and the one that shows sklearn winning on inertia and NMI.

---

## 8. Graph

### 8.1 Mirror adjacency (`_out` and `_in`)

**Why** — citation edges are directed, and the two questions a legal user asks
are mirror images. Without `_in`, "which cases cite X" is an `O(E)` scan; with
it, it is `O(indegree(X))` — 99 entries instead of 3,077 for the seed used in
section 7.3.3. Storage cost: one shared tuple reference per edge.

### 8.2 Edges deduplicated per `(neighbour, class)` pair

The same two cases can be linked by several classes; the class is a filterable
dimension, so both are stored. `neighbors()` deduplicates by
`(neighbour, edge_type)` on demand and `degree()` by neighbour alone, because
"degree" means distinct neighbours.

### 8.3 Hand-written ring-buffer queue

The brief forbids `collections.deque`. A power-of-two ring with `& (n − 1)`
indexing gives `O(1)` amortised enqueue/dequeue with no per-item allocation,
and `_grow` unwraps the live items when it doubles.

### 8.4 Dense `bytearray` visited set over interned ids

**History.** String-keyed hash table → 467 ms on a 10,000-node /
25,087-edge synthetic graph. `_raw_neighbors` generator (no per-node list
materialisation) → 325 ms. Interned ids + bytearray → 86 ms measured during
development; the committed benchmark records **111.41 ms** for the same graph
(`report/tables/summary.json`, `graph.bfs_ms[-1]`), the gap from 86 ms being
run-to-run variance.

**Why it is correct** — ids are assigned lazily by `_index_ids()` on the first
traversal and stay stable, so a bytearray of `len(_index_to_name)` is a valid
membership bitmap for every node.

---

## 9. Candidate restriction on the ML path

### 9.1 `ClusterFilter`: two array reads

**Three designs measured:**

| design | per query | verdict |
|---|---|---|
| `CandidateSet` of the whole probed pool | 1.668 ms | rejected — costs more than the baseline search |
| per-query hash set of members | ~0.45 ms | rejected — the constructor walked all 18,719 documents |
| **cluster mask + doc→cluster array** | **0.0007 ms** | chosen |

The mask is sized to the cluster *count* (not to the largest probed id — an
earlier version threw `IndexError` on a document in an unprobed high-id
cluster), and `contains` guards the mask bound.

### 9.2 Prune by skipping scoring, not by building a second index

**Why** — building a per-cluster index would cost memory proportional to
`k × postings` and would have to be rebuilt whenever the clustering changes.
Skipping a posting before any arithmetic is one array test and keeps a single
index as the source of truth.

---

## 10. Agent

### 10.1 Rule-based intent, word-level matching

**Why not `"cite" in query`** — it fires on "excited" and "precedence". Both are
real lexical queries and both would be routed to a graph that cannot answer
them. Whole-word matching costs one pass and removes the failure; both words are
negative test cases.

**Why not a classifier** — a keyword list cannot be wrong about a query it did
not see, and it is auditable in one screen. The cost is paraphrase ("papers
building on X" → lexical), recorded as a limitation.

### 10.2 Intent beats mode

A relational query runs the graph path even under `--mode ml`. The ML path
prunes *lexical* candidates and cannot answer a relational question; the PATH
line prints both the requested mode and the chosen path so the divergence is
visible.

### 10.3 Two resolution modes for the seed case, both logged

`resolution=query` (id, neutral citation, or unique name fragment) or
`resolution=bm25_top1` (no id found → BM25 over the **legal** index, top case as
seed). The log names which, because a reader must be able to tell whether the
seed was chosen or inferred.

### 10.4 `direction="both"` by default

`06_584` has in-degree 99 and out-degree 0; a forward-only traversal returns the
seed and nothing else. Both directions with BFS deduplication gives each case
once at its minimum hop distance.

### 10.5 The comparison baseline is excluded from `total_ms`

**History.** `run(compare_baseline=True)` executed a second full search inside
the timed region, and the ML path measured 0.54× the baseline *because of the
harness*. Now `total_ms` is the path plus the reasoning/observe overhead, the
comparison baseline is reported separately, and `wall_ms` is the honest
end-to-end number. The corrected result is 1.48× at probe 1.

---

## 11. Front end

### 11.1 `http.server` + JSON, no framework

**Why** — the workbench exists to show measured numbers. A framework would add
a dependency (and an install step) for six endpoints. `ThreadingHTTPServer`
serves them in the standard library.

### 11.2 Loopback `Host` check and a static extension whitelist

Even bound to 127.0.0.1, a browser page on a hostile domain can be made to
resolve an attacker domain to loopback (DNS rebinding). Rejecting non-loopback
`Host` headers blocks the read; the extension whitelist
(`.png .svg .csv .md .json .css .js` under fixed roots) stops a crafted path
reaching `engine.pkl` or source.

### 11.3 Every number on the page comes from the server

The Complexity and Results views read `summary.json` and `tests.json`; the
Search view runs the real agent. There is no hard-coded statistic in the HTML, so
the interface cannot drift from the report — and the Tests view prints how to
generate the file when it is missing rather than inventing results.

### 11.4 Theme tokens defined on bare `:root`, redefined under both dark blocks

A colour defined only inside a `prefers-color-scheme` or `[data-theme]` block
never applies when no attribute is set, which is the default state — the classic
unreadable-artifact bug. Every token is declared on bare `:root` first.

---

## 12. Benchmarks and tests

### 12.1 Nested corpora for the scalability curve

The newsgroups loader sorts by `(category, filename)`, so the 1,000-document
corpus is a strict prefix of the 10,000-document one. The growth is a genuine
corpus-size change, not a different random sample.

### 12.2 `n_init = 10` for sklearn

`n_init = 1` stopped sklearn after 3 iterations at a worse local optimum and
made my implementation look better on NMI — a comparison error in my favour.
The committed run gives sklearn its default and reports that it wins on inertia
(3,780.1 vs 3,789.9) and NMI (0.3539 vs 0.3458), while mine is 1.18× faster.

### 12.3 Memory measured with a floor

`tracemalloc` around construction only, and the reference peak is
`max(dict_peak, 1.0)` so a rounding artefact cannot print a negative delta
(the earlier output said `-6144 KiB`, which is nonsense).

### 12.4 Tests are oracles, not smoke tests

The hash table is compared with `dict` under random insert/delete sequences;
`TopKHeap` with a full sort for every `k` in `{1, 3, 10, 500, 1000}`; BM25 with
`rank_bm25` at the same tokenisation; BFS on cycles, disconnected graphs, depth
limits and both directions; K-Means on synthetic blobs; and the architecture
rule by AST parsing. 232 tests, all passing.
