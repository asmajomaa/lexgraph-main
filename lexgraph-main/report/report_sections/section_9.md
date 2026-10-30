# 9. Conclusion

## 9.1 What the project demonstrates

Three fields, one engine, every number measured:

| field | what was built | what it measured |
|---|---|---|
| **IR** | polynomial-hash table with chaining, inverted index, Okapi BM25, bounded top-K heap | 3,215× faster than a naive scan at 18,719 documents (peak 3,215× at 18,719); 99.6 % top-10 agreement with `rank_bm25` |
| **ML** | TF-IDF vectoriser (top-5,000 terms, L2), K-Means with k-means++, purity/NMI/ARI from scratch | purity 0.3141, NMI 0.3078, ARI 0.1081 at `k = 20`; inertia matching sklearn to 0.26 % on a 4,000-document sample |
| **Graph** | typed directed citation graph with forward and reverse adjacency, ring-buffer BFS, shortest path, components | 3,890 cases / 3,077 edges / 17 edge classes; BFS linear in `V + E` from 0.64 ms to 111.41 ms across a 100× size increase (101× in `V + E`) |
| **Agent** | rule-based ReAct loop routing to three paths, every step logged | relational answers in ~20 ms; ML pruning 1.23–1.48× faster with 45–70 % recall loss |

The graded constraint — no `dict`-as-index, no `heapq`, no `networkx`, no
`scikit-learn`, no `numpy` in the engine — is enforced by a test that parses
every module's AST (`tests/test_purity.py`), not by a promise in a README. The
full suite is 232 tests, all passing.

## 9.2 What I learned building it

**The first measurement you take is usually wrong, and the wrongness has a
shape.** My ML path measured 17× faster, then 6.6×, then 0.54× *slower* than
the baseline, on the same code. Each number was produced by a harness bug, not
a model change: the 17× timed only the scoring loop and called it the path; the
0.54× timed the baseline *inside* the ML path's own window. Only the third
version — with `run()` separating the path's cost from the comparison
baseline, and reporting `wall` separately — gives a defensible 1.48×. A
benchmark that flatters the thing it measures is the easiest kind to write and
the least useful.

**A proof tells you where the constant factors are, not where the time goes.**
Section 4.4.1's `O(1)` argument was correct on the first try and the table was
still too slow to matter; the fix was not the complexity but replacing a
`%` with a mask and a tombstone with a splice. Conversely, the BFS was already
`O(V + E)` and still took 467 ms on 10,000 nodes until the visited check
stopped hashing strings.

**An optimisation that is architecturally right can still be a net loss.** The
cluster filter is `O(k)` per query and replaced an `O(N)` candidate set — a
strict asymptotic win — and the first version still cost ~0.9 ms per query
because it *materialised* the pool. Only when it became two array reads
(0.0007 ms) did pruning beat the thing it was accelerating. "Faster in theory"
is a hypothesis; the stopwatch is the experiment.

**Report the inconvenient number.** The clustering is mediocre (purity 0.31),
and that single number explains the recall loss in the pruning experiment. It
would have been easy to lead with the 1.48× speedup and bury the 29.6% recall;
the project is more useful — and more honest — with the two numbers in the same
table.

**A pure-Python structure is competitive when it changes the algorithm, not
when it imitates a C one.** My hash table loses to `dict` by 136.4× because it is
Python objects doing what C does better. My BM25 beats `rank_bm25` by 8.8× and
my K-Means beats sklearn on time — because they are not imitations but different
data layouts (inverted postings, sparse-dense distance). The lesson is about
where Python's constant factor is worth paying and where it is not.

## 9.3 Limitations

**No semantic understanding.** Both lexical paths are exact-term matchers with
a light Porter stemmer. "Goaltender" does not match `goalie`; "how do I stop a
puck" does not match a post about saves unless the words coincide. The
vectoriser is TF-IDF over the same tokens, so the *clustering* has the same
blindness. Section 7.5's recall numbers are partly a symptom of this.

**A light stemmer, not a full one.** `core/tokenizer.py` implements Porter
steps 1a, 1b and a trimmed step 4; steps 2, 3 and 5 are omitted. `running →
run`, `applied → apply`, `studying → study` all work (pinned by tests), but
derivational pairs like `generalisation`/`generalise` do not collapse. That is a
deliberate trade-off — the full algorithm is ~200 lines and the measured
improvement on this corpus is small — but it is a limitation.

**Pruning loses recall, and the report does not hide it.** 29.6 % / 46.9 % /
55.0% of the baseline's top-10 at probe 1/2/3, for 1.48× / 1.34× / 1.23×
speedup. There is no setting in the experiment that is both meaningfully faster
and recall-safe, and the honest reading is that cluster pruning on TF-IDF
vectors is not worth its recall cost for a top-10 user.

**The clustering is weak.** Purity 0.3141 on 20 classes means roughly a third
of documents would get the right label under majority voting. The PCA scatter
shows why: the first two components explain 0.61 % and 0.41 % of variance, so
the "clusters" are slices of one blob rather than separated islands.

**K-Means did not converge.** 60 iterations, `max_iter`, inertia still moving
in the fourth decimal. The inertia curve (7.6) suggests very little was left to
gain, but "stopped at the budget" is the accurate description.

**The citation graph is a local slice.** 22,381 extracted citation records
point at cases outside the 3,890-case corpus and are dropped; 1,526 of 3,890
nodes are isolated; average degree is 0.791. Relational queries only work
within that slice, and the low density means many cases have no relational
answer at all.

**The relational router is a keyword list.** "Papers building on X" routes to
the lexical path, because it contains no word from
`config.RELATIONAL_KEYWORDS`. It cannot be wrong about a query it did not see,
which is what makes it auditable, but it also cannot generalise.

**The build is slow.** 193.00 s, of which K-Means is 93.74 s and the legal
index 48.54 s. Acceptable for a one-off build, not acceptable as a per-startup
cost — which is exactly why the pickle cache exists (load is a single read,
219.4 MB).

**Single-user assumptions.** The web server binds loopback, rejects
non-loopback `Host` headers and serves a static whitelist; there is no
authentication, no concurrency control over the engine object (queries read
shared state), and no rate limiting. It is a demonstration surface, not a
production service.

**The pickle cache is not portable across code changes.** Its format is
documented in section 5.10 and the README; a fingerprint covering the corpus
size, vocabulary size, `k`, `max_iter`, seed and tokenizer version invalidates
it, but a change to the *on-disk schema* needs `CACHE_VERSION` bumped by hand.

**Windows-only verification.** Every number in this report was produced on
Windows 11 with CPython 3.10. The code is platform-neutral (no path separators
hard-coded outside `os.path`, no OS-specific APIs) but the timings are not, and
they have not been reproduced on Linux or on another Python version.

## 9.4 Future work

**Dense vectors instead of TF-IDF.** The single change with the largest
expected effect on every weak number in this report. Replacing the 5,000-term
sparse vectors with sentence embeddings (a few hundred dense dimensions) would
(a) make K-Means clusters correspond to topics rather than shared vocabulary,
(b) raise purity and NMI, and (c) — the point — make pruning recall-safe,
because a query and its true answers would land in the same cluster. The
pruning architecture (`ClusterFilter`, two array reads) would not change at all;
only the vectoriser and the centroid representation would.

**HNSW over the cluster filter.** K-Means pruning is a coarse partition; an
approximate-nearest-neighbour index over document vectors would give a
per-query candidate set with a *tunable* recall target (`ef_search`) instead of
a fixed "two clusters". That is the standard successor to cluster pruning in
large-scale retrieval, and it would let the ML path expose recall as a
parameter rather than a consequence.

**A learned query router.** The keyword list is auditable but cannot recognise
"papers building on Antipova". A small classifier over the query — or an LLM
prompt — would generalise; the work is to keep the *trace* as honest as it is
now, i.e. to log why a path was chosen and what confidence it had.

**An LLM-based reasoning agent.** The current loop is fixed: Reason classifies,
Act dispatches, Observe reports. An LLM in the Reason step could plan a
multi-step retrieval ("first find the case, then list what cites it, then rank
those by BM25 against `migration review`"), and the `[AGENT][REASON]` log
format already accommodates that. The risk is the one this project was built to
avoid: a router that cannot be audited. A useful hybrid is an LLM proposer with
a rule-based checker, logging both.

**A full Porter stemmer, or a learned one.** Cheap to add, measurably better on
derivational pairs, and it would raise the BM25 agreement with `rank_bm25`
above 99.6 % (the remaining 0.4 % is largely stemmer mismatch).

**Weighted BM25F over typed edges.** The legal index already boosts
catchphrases 3×; extending that to field weights over catchphrase, sentence and
citation text — the standard BM25F formulation — is a natural next step and
would be a few lines given the loader already separates the fields.

## 9.5 Closing

The project's headline is not the 3,215× speedup over a naive scan — that is
what an inverted index is *for*. It is that the speedup, the recall loss, the
cluster quality and the 136.4× hash-table penalty are all measured from the same
running code, recorded in `report/tables/`, and reproduced by the committed
scripts. A reader can disagree with any conclusion here by re-running a
benchmark; that is the property the three-fields brief was really asking for.

```bash
python main.py --build                    # 193.00 s, once
python main.py --query "hockey playoffs goalie" --mode ml --compare
python main.py --query "which cases cite 06_584"
python main.py --serve                    # workbench on 127.0.0.1:8000
python -m benchmarks.run_all              # every figure and table
pytest                                    # 232 passing
```
