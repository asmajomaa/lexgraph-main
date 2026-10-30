# 2. The problem solved by the algorithm

## 2.1 What information retrieval is

An **information retrieval (IR) system** answers a user's information need,
expressed as a short query, with a ranked list of documents from a collection
it has never seen the query for. Three things separate IR from a database
query:

1. **The match is partial.** A document need not contain every query word, and
   the best answer may contain none of them literally.
2. **The output is ranked, not filtered.** Saying "these 400 documents are
   relevant" is useless; the user reads the top ten.
3. **Relevance is graded and uncertain.** A document is *more* or *less* about
   the query, and there is no ground truth to check against at query time.

The classical architecture that handles this is the **inverted index**: for
each term, a list of `(document, term frequency)` postings. A query walks the
posting lists of its own terms and accumulates a score per document, touching
only the documents that contain at least one query word. The engine here builds
that structure on its own hash table, and the postings live under the term key
in `PostingsTable`.

## 2.2 Why plain keyword matching is insufficient

The naive alternative — scan every document, count how many query words appear
— fails in four measurable ways.

**(a) It does not scale.** A full scan is `O(N · L)` per query, where `L` is
the average document length. Measured on this corpus (`benchmarks/bench_scalability.py`,
1000→18,719 documents, five fixed queries, mean of five runs):

| corpus | naive full scan | inverted index + BM25 | speedup |
|---|---|---|---|
| 1,000 | 240.268 ms | 0.147 ms | 1635× |
| 2,000 | 396.835 ms | 0.193 ms | 2055× |
| 5,000 | 879.477 ms | 0.364 ms | 2417× |
| 10,000 | 2058.658 ms | 0.751 ms | 2742× |
| 18,719 | 4,648.113 ms | 1.446 ms | 3,215× |

The naive curve grows linearly in `N` (4105.7/240.3 = 17.1 against an 18.7×
corpus growth); the inverted index grows far more slowly because a query only
touches the postings of its own terms. At the full corpus the difference is
three orders of magnitude, and the speedup peaks at 3,215× around 18,719
documents — at 18,719 the postings of the five fixed queries are long enough
that the constant factor rises faster than `N`.

**(b) It ignores term rarity.** "the" appears in almost every document and
carries no signal; "goalkeeper" appears in a handful. A count-based score ranks
a document that repeats "the hockey" above one that mentions "goalkeeper" once.
BM25 weights each term by inverse document frequency:

```
idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
```

At the full corpus, `hockey` has `idf = 4.162`, `playoffs` `4.966` and `goalie`
`5.839` (from `python main.py --explain "hockey playoffs goalie" --doc 3870`)
— the rarer the term, the more a match is worth, which is exactly the ordering
a reader expects.

**(c) It ignores document length.** A long post repeats every word it talks
about; without normalisation it wins by bulk. BM25's denominator
`tf + k1(1 - b + b·dl/avgdl)` shrinks each contribution as the document grows
past `avgdl = 101.46` tokens.

**(d) It cannot answer relational questions at all.** "Which cases cite
06_584" is not a bag-of-words question: the answer is defined by the citation
graph, and the relevant documents may share no vocabulary with the query. A
keyword engine returns the case itself and stops. The relational path exists
because IR over linked documents is a graph traversal problem, and the whole
point of a *hybrid* engine is to route by intent rather than to pretend one
model fits every question.

## 2.3 The specific problem this project solves

Given two heterogeneous corpora, answer three kinds of question with one
engine and one agent:

1. **Lexical** — "hockey playoffs goalie" → rank 18,719 newsgroup posts by
   BM25, and do it fast enough (a **1.493 ms** mean over 100 sampled queries,
   section 7) that the K-Means pruning experiment is meaningful rather than
   noise-dominated.
2. **Lexical, accelerated** — the same, but first ask K-Means which clusters
   the query is near and score only those. The question is not "is clustering
   accurate" but "how much of the corpus can be skipped, and what does that cost
   in recall". Measured answer: probing one cluster scores **165 documents
   (0.9 % of the corpus)** and is **1.48× faster**, but returns only **29.6%**
   of the baseline's top-10. Probing two costs 1.34× speed and recovers
   **46.9%**; three costs 1.23× and recovers **55.0%**. Pruning trades recall
   for latency, exactly as the theory predicts, and the report states the loss
   rather than hiding it.
3. **Relational** — "which cases cite 06_584" → BFS two hops over the typed
   citation graph from the seed case, in both directions, returning 194 cases
   in 27.6 ms on a cold engine (16.7 ms of that is the BFS; the first query
   also pays a one-off index build) — console transcript in section 7. On a warm
   engine the same query is 6.5 ms, because the case-id map is built once and
   reused.

## 2.4 Why the structures must be hand-built

Writing the hash table, heap, graph, vectoriser and K-Means ourselves is not
nostalgia. It makes three things checkable that a library hides:

* **The cost is visible.** `HashSetTable.collisions` (lifetime inserts) and
  `.chain_collisions` (entries sharing a bucket in the finished table) are
  public counters, as is `.probes`; `BM25.scored_documents` records exactly how
  many postings a query touched; the agent prints "scored 413 documents (2.2%)"
  because the scorer knows. With `dict` there is no collision count to report,
  and no way to tell a well-distributed vocabulary from a clumped one.
* **The cost is tunable.** The 0.75 load factor, the `k1`/`b` values, the
  vocabulary cap and the probe count are all in `config.py` and all change
  measured behaviour.
* **The comparison is fair.** Every reference implementation in section 8 runs
  on *the same tokenised input* as mine — the tokenizer object is passed to
  `rank_bm25` and to sklearn's `TfidfVectorizer` — so the differences in the
  tables are implementation, not data.

The price is performance, paid openly: a pure-Python chain-based table is
136.4× slower than `dict` for bulk insertion and 13.7× slower
for lookup (section 8; that insert ratio moves between 119× and 136× across six
recorded runs on this machine, because wall-clock timings of a 2-second loop
follow machine load), and my BFS is 11.8× slower than
`networkx`'s C-backed adjacency walk (12.038 ms against 1.023 ms). Section 9 discusses whether that price is
worth paying for the transparency it buys.
