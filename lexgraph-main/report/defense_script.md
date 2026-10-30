# Defense script — 5 minutes

Every number here appears in `report/report_sections/section_7.md` or section 8
and is traced to `report/tables/summary.json` or a printed transcript. Nothing
is rounded up or invented. Times are for the machine the benchmarks ran on
(Windows 11, CPython 3.10).

**Length check.** The spoken text is **733 words** — 4 min 53 s at 150 words per
minute, 5 min 14 s at 140. The timestamps below are the measured block ends at
140 wpm, the slower pace, so a faster delivery buys pauses rather than overrun.

---

## 0:00 — What it is (46 s)

"LEXGRAPH-RAG is a hybrid retrieval engine over two real corpora: 18,719 Usenet
posts from 20 Newsgroups and 3,890 Australian court cases from the Legal Case
Reports set.

Three fields, one system. Information retrieval: an inverted index and Okapi
BM25 from scratch. Machine learning: TF-IDF and K-Means, used to prune the search
space. Graph theory: a typed citation graph with breadth-first search.

The constraint is that the hash table, the heap, the graph, the vectoriser and
K-Means are all mine — no `dict` as index, no `heapq`, no `networkx`, no
scikit-learn, no numpy. A test parses every engine module's syntax tree and fails
on a forbidden import."

## 0:46 — Why keyword matching is not enough (41 s)

"The naive alternative is to scan every document per query. I measured it: at
1,000 documents that costs 267 ms; at 18,719 it
costs **4,648 ms**. My index answers in **1.446 ms** —
**3,215×** faster, and the gap widens with corpus size
because a query only touches the postings of its own terms.

But counting words ignores term rarity and length. My `--explain` output on a
real post shows it:

```
N=18719  dl=22  avgdl=101.4558  k1=1.5  b=0.75
hockey      tf=1  idf=4.162308  contribution=6.427486
playoffs    tf=1  idf=4.965975  contribution=7.668518
goalie      tf=1  idf=5.839147  contribution=9.016882
total = 23.112887
```

'Goalie' outweighs 'hockey' because it is rarer, and the document is 22 tokens
against an average of 101, so length normalisation boosts it."

## 1:27 — The structures and the proofs (60 s)

"Four structures carry the project.

The hash table: polynomial hash, base 31, modulo 2⁶¹−1, Horner's rule, separate
chaining, resize at load factor 0.75. Expected chain length is 1 + α, α capped at
0.75, doubling gives amortised O(1). It is also checked: 21,912 collisions
against 20,917 predicted, mean chain 0.428, longest 6.

The heap is a bounded min-heap of the best-so-far — the worst survivor sits at the
root, so one comparison rejects almost every candidate. O(N log K) time, O(K)
space.

BFS keeps forward and reverse adjacency, because 'what X cites' and 'what cites X'
are mirror questions. Its queue is a ring buffer and its visited set a bytearray —
that took BFS from 467 to 86 ms.

K-Means uses the sparse-dense identity, so a distance costs the number of
non-zeros instead of 5,000 coordinates."

## 2:27 — The agent (48 s)

"The agent is a rule-based ReAct loop, not an LLM. Reason classifies the intent,
Act dispatches to one of three paths, Observe reports what happened, Answer gives
one sentence. Every step prints a log line, and the logs are what the report
quotes.

One rule matters: intent beats mode. 'Which cases cite 06_584' takes the graph
path even under `--mode ml`, because pruning lexical candidates cannot answer a
relational question.

A real run resolved the seed from the query, walked two hops both ways, and
reached **194 cases in 16.7 ms**. The seed has in-degree 99 and out-degree 0, so
without reverse adjacency the answer would have been the case itself."

## 3:15 — The honest ML result (55 s)

"This is the part I want to be judged on.

K-Means on 5,000-term TF-IDF gets purity **0.3141**, NMI **0.3078**, ARI
**0.1081** at k = 20. Mediocre numbers, reported as mediocre. The scatter plot
shows why: the first two principal components explain 0.61 % and 0.41 % of
variance.

So pruning follows. Over 100 queries: probing 1 cluster is **1.48×** faster
but recovers only **29.6%** of the baseline's top-10; probing 2 gives **1.34×**
and **46.9%**; probing 3 gives **1.23×** and **55.0%**. Documents scored fall from
1,082 to 165 per query, yet wall-clock barely moves, because the baseline's cost
is walking posting lists.

Getting that honest took three tries: the first said 17×, timing only the scoring
loop; the second said 0.54×, because it timed the comparison search inside the ML
window."

## 4:10 — Comparison with existing libraries (42 s)

"Against the libraries a practitioner would use, identical inputs and
tokenisation: my hash table is **136.4× slower** to insert than `dict` and uses
**1.80×** the memory — Python objects against C open addressing. My heap is
**4.1× slower** than `heapq` with **identical output**; BFS is **11.8× slower**
than networkx.

But BM25 is **8.8× faster** than `rank_bm25` at **99.6 %** top-10 agreement,
because it scores densely and I score only the query's postings. And K-Means is
**1.18× faster** than sklearn on 4,000 documents, inertia 3,789.9 against 3,780.1
— sklearn wins slightly on quality, because it restarts ten times."

## 4:52 — Limitations and close (25 s)

"Limitations: no semantic understanding, a light stemmer, weak clusters, 22,381
citations pointing outside the corpus, a slow build, a single-user server.

The change with the largest expected effect is dense vectors: it would fix the
clustering, and clustering caps the pruning recall. The architecture would not
change at all.

Thank you — I can run any of this live."

---

## Appendix: commands to have ready

```bash
python main.py --query "hockey playoffs goalie" --mode baseline --k 5
python main.py --query "hockey playoffs goalie" --mode ml --compare
python main.py --query "which cases cite 06_584"
python main.py --explain "hockey playoffs goalie" --doc 3870
python main.py --serve          # workbench: Search / Tests / Complexity / Results
pytest
```

## Appendix: numbers to have memorised

| claim | number |
|---|---|
| naive scan at full corpus | 4,648.113 ms |
| inverted index + BM25 | 1.446 ms |
| speedup | 3,215× (peak 3,215× at 18,719) |
| BM25 top-10 agreement with `rank_bm25` | 99.6 % |
| ML speedup / recall at probe 2 | 1.34× / 46.9% |
| K-Means purity / NMI / ARI | 0.3141 / 0.3078 / 0.1081 |
| hash insert vs `dict` | 136.4× slower |
| heap vs `heapq` | 4.1× slower, same output |
| BFS vs networkx | 15.1 ms vs 1.0 ms |
| relational walk from `06_584` | 194 cases in 16.7 ms |
| legal graph | 3,890 cases / 3,077 edges / 1,526 isolated |
| build | 193.00 s |
| tests | 232 passing |
