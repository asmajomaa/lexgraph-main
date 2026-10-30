# Complexity notes

This document states, for every graded component, what the code does, the cost
it pays, and *why* that cost is what it is. The per-function table is generated
from the source by `python -m report.generate_report` (it parses each
docstring, so the table cannot drift away from the code); the proofs below are
the reasoning behind the table.

Notation: `N` documents, `V` vocabulary size, `P` postings visited by a query,
`K` requested results, `L` key length, `n` items in a heap, `k` clusters,
`nnz` non-zero coordinates of one sparse vector, `M` vector dimension,
`T` K-Means iterations, `V_g`/`E_g` graph nodes/edges.

---

## 1. Hash table (`structures/hash_table.py`)

**Structure.** An array of `capacity` bucket heads (a power of two), each head
pointing into a hand-written singly linked chain of `_Entry` nodes. A key maps
to a bucket through a polynomial rolling hash, base 31, modulo `2^61 - 1`,
evaluated with Horner's rule. The table doubles its capacity and rehashes when
`size / capacity` exceeds 0.75.

### Claim: insertion is O(1) amortised

Hashing a key of length `L` costs `L` multiply-add-mod operations, so the
per-operation cost is `O(L)` for a fixed key set (treat `L` as a constant for
English vocabulary, where the mean token length is a handful of characters).

Let `alpha = n / capacity` be the load factor. Under the simple uniform
hashing assumption each of the `n` keys is equally likely to land in any of the
`capacity` buckets, so the chain length at a bucket is
`Bin(n, 1/capacity)` and every walk costs `1 + n/capacity` expected steps.
Resizing keeps `alpha <= 0.75` at all times *after* the triggering insert, so
the expected chain walk is at most `1.75` steps — a constant.

The resize itself copies `n` entries and costs `O(n + capacity)`. Doubling
means that after `m` inserts the total resize work is
`sum_{i=0..log m} 2^i = 2m - 1`, i.e. `O(m)` over `m` insertions, so each
insert is charged `O(1)` amortised. This is the standard doubling argument; the
0.75 threshold gives a constant factor of 4/3 on the array size relative to the
minimum, which is why both the array and the chains stay small.

**Measured sanity check** (`report/tables/hash_collision_metrics.txt`, produced
by `python -m report.hash_collision_metrics`): the newsgroups postings table
holds 112,184 terms in 262,144 buckets (load factor 0.428), the longest chain is
6 nodes and the mean over *all* buckets is 0.428. The **final table** has 21,912
entries sharing a bucket against 20,917 predicted by the uniform model above
(ratio 1.048) — the polynomial hash is behaving close to ideal, and it is not
the reason my table loses to `dict` (that is constant factors: Python-level node
objects versus C-level open addressing).

Do not confuse that with the table's *lifetime* counter, which reads 83,222 for
the same keys inserted one `put` at a time: each of the 14 doublings re-inserts
every entry, so the counter keeps counting keys a second and third time. Both
numbers are honest; they answer different questions (table shape versus build
work).

**Worst case.** If every key collides into one chain, `put`/`get` degrade to
`O(n)`. The honest statement is average `O(1)`, worst `O(n)`; a table without
randomised hashing can always be attacked with deliberately colliding keys.

**Space.** `O(n + capacity)`, and because `alpha <= 0.75` implies
`capacity <= (4/3)n` after any resize, that is `O(n)`.

### Design choices

| choice | alternative | why |
|---|---|---|
| base 31 | base 2, base 131 | 31 is prime, keeps `h*31` inside one machine word for realistic token lengths, and separates anagrams well (`"ab" = 97*31 + 98 = 3105` vs `"ba" = 98*31 + 97 = 3135`). |
| modulus `2^61 - 1` | `2^63`, `2^32` | A Mersenne prime near the top of the 64-bit signed range: no wrap-around into big-int arithmetic in CPython, still a genuine prime so the polynomial hash has good distribution. |
| separate chaining | open addressing | Deletion is a pointer splice rather than a tombstone; and chain length degrades gracefully if the hash is clumpy, which matters because the keys are English words with a skewed length distribution. |
| power-of-two capacity | prime capacity | `hash & (capacity - 1)` replaces `hash % capacity`; the modulo is already taken modulo a prime inside `polynomial_hash`, so the low bits are well mixed. |
| 0.75 load factor | 0.5, 0.9 | Textbook sweet spot: expected probes `1 + alpha/2` for chaining stays near 1.4 while resizes stay rare. Lower wastes memory in a 1.4 M-posting index; higher makes chains visibly longer. |

---

## 2. Bounded top-K heap (`structures/min_heap.py`)

### Claim: selecting the top K of N candidates costs O(N log K) time and O(K) space

`TopKHeap` is a binary min-heap of capacity `K` that stores the *worst* of the
current best-`K` set at the root, ordered by the total order "higher score
first, then lower `doc_id`" (`_better`).

For each candidate:

* if the heap is not full, push it — one `_bubble_up`, `O(log K)`;
* if it is full, compare against the root in `O(1)`; when it loses, nothing
  happens; when it wins, pop the root (`_sink_down`, `O(log K)`) and push the
  candidate (`_bubble_up`, `O(log K)`).

Every candidate therefore costs `O(1)` in the losing case and `O(log K)` in
the winning case, so `N` candidates cost `O(N log K)` in the worst case and
`O(N + W log K)` where `W` is the number of winners — for a random stream where
only `O(K log N)` items ever enter, closer to linear.

The heap never holds more than `K` items, so the space is `O(K)` regardless of
`N`. Sorting the candidates instead would cost `O(N log N)` time and `O(N)`
space — a factor `log N / log K` worse in time and `N/K` worse in memory, which
matters when `N = 1,400,000` postings and `K = 10`.

`get_top_k()` drains the heap (`K` pops, `O(K log K)`) and reverses, giving
descending order; ties break on the lower `doc_id` inside `_better`, so the
output is deterministic for a fixed input, which is what makes the test suite's
comparison against `heapq` meaningful.

---

## 3. Citation graph and BFS (`structures/graph.py`)

**Structure.** Two custom hash tables: `_out` maps a case id to a list of
`(target, edge_type)` pairs, `_in` maps it to `(source, edge_type)` pairs. Both
are needed because "which cases does X cite" and "which cases cite X" are the
two questions the agent must answer, and a citation graph is directed.

### Claim: BFS costs O(V_g + E_g)

`bfs_traverse` runs a textbook breadth-first search with three additions:

1. **A hand-written ring-buffer queue** (`_CircularQueue`, capacity a power of
   two, `O(1)` amortised enqueue/dequeue) instead of `collections.deque`, which
   the brief forbids.
2. **A dense `bytearray` visited set indexed by an interned node id**, so
   marking and testing a node is one array index rather than a string hash.
3. **A depth cut-off**: nodes at `depth == max_depth` are dequeued but their
   neighbours are not scanned.

Each node is enqueued exactly once (the visited bit is set at enqueue time, not
at dequeue time — that is what keeps the queue free of duplicates), so the
outer loop runs at most `V_g` times. Each edge is examined exactly once, when
its source node is dequeued, so the inner loop runs at most `E_g` times. With
`O(1)` work per edge — one `node_ids.get` and one bytearray read — the total is
`O(V_g + E_g)`.

The depth cut-off only *helps*: with horizon `h` the traversal is bounded by the
nodes within `h` hops, but the worst case over all `h` is still `O(V_g + E_g)`.

Interning is what makes the `O(1)` claim real. Before it, each visited test
hashed a case id string (`"06_584"`, `"07_1318"`; six or seven characters,
mean 6.4 over the 3,890 real ids).
The bytearray version replaces that hash with one list index, and the
`_raw_neighbors` generator avoids materialising a neighbour list per node.
During development the 10,000-node / 25,087-edge synthetic traversal measured
467 ms before either change, 325 ms after the generator alone, and 86 ms with
the bytearray; the committed benchmark run recorded **111.41 ms**
(`report/tables/summary.json`, `graph.bfs_ms[-1]`), the difference from 86 ms
being run-to-run variance on a busy desktop.

**Space** is `O(V_g)` for the visited bytearray, the depth/parent arrays and
the queue — all sized by node count, never by edge count.

**Why reverse adjacency.** A citation edge is directed: case A cites case B,
not the reverse. The interesting query, "which cases cite A", is exactly the
reverse direction. Storing only the forward list would force a full scan of
every edge to answer it (`O(E_g)` per query); storing both makes it
`O(indegree(A))`. The cost is one extra list per node, i.e. `O(E_g)` space, and
one extra append per edge at build time — a fair trade for turning a linear scan
into a constant-in-corpus operation.

`shortest_path` is BFS with a parent map and costs the same `O(V_g + E_g)`;
`connected_component_size` is BFS restricted to the undirected view and is
`O(V_g + E_g)` on the component, not on the whole graph.

---

## 4. Inverted index and BM25 (`core/inverted_index.py`, `core/bm25.py`)

**Layout.** `PostingsTable` (a `HashSetTable`) maps `term -> [(doc_id, tf), ...]`
in insertion order, i.e. ascending `doc_id`, which lets the scorer do a
merge-style intersection later if it wants one. `doc_lengths`, `labels`,
`sources`, `external_ids`, `titles` and `snippets` are parallel lists indexed by
`doc_id`. `total_length` is accumulated at build time so `avgdl` is `O(1)`.

### Claim: a query costs O(sum_t postings(t) + P log K)

`BM25.search` (`core/bm25.py:161`) works term-at-a-time:

1. `_prepare_query` tokenises the query (`O(len(query))`), then for each unique
   term looks up `df` (`O(1)` expected from the postings length) and computes
   `idf` (`O(1)`).
2. For each term, every posting is visited once: `sum_t |postings(t)|` visits.
   Each visit is `O(1)`: one candidate test, one `term_score`, one array add.
3. The score array is `N` floats, allocated once per query and reused by the
   caller in benchmarks.
4. The heap pass is `O(P log K)` over the `P` documents that actually scored
   above zero.

So the time is `O(sum_t |postings(t)| + P log K)` and the space is
`O(N + K)` — the dense score array necessarily touches every doc id, which is
why the pruned path tries to make `P` (not `N`) the driver. That is an honest
limitation, not a win: pruning reduces the number of *scoring* operations and
the size of the heap input, not the allocation of the `N`-float array.

The `RestrictedView` used by the ML path returns the **global** document
frequency. Two reasons: first, computing candidate-local df would mean walking
every posting of every query term per query, which would erase the pruning
saving entirely; second, using the same idf on both paths is what makes the
Recall@10 comparison meaningful — a different idf would change the ranking for
reasons unrelated to pruning.

---

## 5. Vectoriser and K-Means (`ml/vectorizer.py`, `ml/kmeans.py`)

### Claim: `fit` costs O(n · k · nnz · T)

Assignment (the inner loop) computes, for each of the `n` documents, the
squared distance to each of the `k` centroids.

Naively that is `O(k · M)` per document — `M = 5,000` coordinates, almost all
zero. The implementation uses the sparse-dense identity

```
||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
```

where `||x||^2` is cached on the `SparseVector`, `||c||^2` is recomputed once
per iteration (`O(k·M)`), and `x·c` is a merge walk over the `nnz(x)`
non-zeros. So the assignment costs `O(n · k · nnz)` instead of `O(n · k · M)`.
With `M = 5,000` and `nnz ≈ 60` for a newsgroup post, that is a factor of ~80,
and it is the only reason a pure-Python Lloyd's algorithm finishes at all.

The update step accumulates each document into its cluster's dense sum
(`O(n · nnz)`, not `O(n · M)`), then divides by the count (`O(k · M)`). An
empty cluster is reseeded to the point farthest from its own centroid, which
costs `O(n)` with the cached norms.

Per iteration: `O(n·k·nnz + n·nnz + k·M)`. Over `T` iterations:
`O(T · n · k · nnz)` for the dominant term, which is the `O(n·k·d·iters)` in
the brief with `d` specialised to the non-zero count.

**Space.** `O(k·M)` for the centroids (`20 × 5,000 × 8 B = 800 kB`), `O(n)` for
the labels and `O(n·nnz)` for the vectors themselves. Dense centroids are the
right choice because *every* centroid is compared to every document — a sparse
centroid representation would trade a memory win for a constant-factor loss on
the hot path.

`predict_top_n` costs `O(k · nnz)` for the dots plus `O(k log k)` for the sort
— with `k = 20` that is trivial, so the ML path's overhead is dominated by the
`O(k · nnz)` walk into the posting lists of the query's few terms.

`TFIDFVectorizer.fit` counts document frequencies over the corpus
(`O(total tokens)`), selects the top `M` terms by `(-df, term)` — the term
tie-break makes the vocabulary deterministic across runs — and computes the
smoothed idf once per term: `O(V log V)` for the selection.

`TFIDFVectorizer.transform_text` is `O(len(text) + u log u)` for `u` unique
in-vocabulary terms, the log coming from sorting the sparse pairs, which
`SparseVector` requires so that `dot` can be a linear merge walk rather than a
hash lookup per coordinate.

---

## 6. Clustering metrics (`ml/evaluation.py`)

All four are single-pass over the contingency table `C` (clusters × classes),
sized `k × c` (here `20 × 20`):

* `contingency_table`: `O(n)`.
* `purity`: `O(k·c)` — one max per row.
* `nmi`: `O(k·c)` after the row and column entropies.
* `adjusted_rand_index`: `O(k·c)` using the pair-count form
  `sum_ij C(n_ij, 2)`, `sum_i C(a_i, 2)`, `sum_j C(b_j, 2)` and `C(n, 2)`.

None of them ever builds an `n × n` matrix, which is what a naive ARI
implementation does.

---

## 7. The agent (`core/agent.py`)

`run` is `reason` + `act` + `observe` + `answer`:

| step | cost |
|---|---|
| `reason` | `O(len(query))` — word-split and a membership test against 14 keywords |
| `act` (baseline) | `O(P log K)` — BM25 over the whole corpus |
| `act` (ml) | `O(k·nnz + P_candidate log K)` — vectorise, probe two clusters, score only the postings inside them |
| `act` (relational) | `O(V_g + E_g)` bounded by the depth horizon |
| `observe` | `O(K)` — materialise the `K` rows |
| `answer` | `O(K)` — read the top title and label |

`ClusterFilter.contains` is two array reads — `cluster_of[doc_id]` then
`mask[cluster]` — so the per-posting overhead of pruning is a constant, and the
mask is one byte per cluster (`k = 20` bytes), not a per-query candidate set.
Building a `bytearray` of every document in the two nearest clusters (the
obvious approach) measured **1.668 ms** per query on the built engine; this
filter costs **0.0007 ms**.

---

## 8. Benchmarks and test harness

`benchmarks/` and `tests/` are outside the graded core and are allowed to
import `dict`, `heapq`, `numpy`, `sklearn`, `networkx` and `matplotlib`: they
exist precisely to *check* the core against those libraries. `structures/`,
`core/` and `ml/` import none of them, and use no `dict`/`set` as a lookup
table either — `tests/test_purity.py` parses every engine module with `ast`,
fails on a forbidden import, fails on any empty `{}`/`set()`/comprehension
built inside an engine function, and asserts behaviourally that the inverted
index, both adjacency tables, the vectoriser vocabulary, BM25's query maps and
the contingency table are the project's own structures.
