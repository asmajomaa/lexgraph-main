# Public API: input, output, time and space

Generated from the source by `report/generate_report.py`; each row is a
public function with the complexity stated in its docstring. Private
helpers (`_bubble_up`, `_sink_down`, `_resize`, ...) are covered by
`docs/complexity.md`.

## structures/hash_table.py -- hash table and postings

| function | input | output | time | space |
|---|---|---|---|---|
| `polynomial_hash(key: str, base: int = HASH_BASE, modulus: int = HASH_MODULUS)` | key: The string to hash. base: Multiplicative base (default 31). modulus: Prime modulus (default ``2**61 - 1``). | A non-negative integer in ``[0, modulus)``. | O(L) where L = ``len(key)`` | O(1) |
| `hash_key(key: Any)` | key: A ``str``, an ``int`` or a (non-empty) tuple of those. | A non-negative integer in ``[0, modulus)``. | O(len(key)) for a string, O(1) for an integer, O(number of elements) for a tuple | O(1) |
| `HashSetTable.put(key: str, value: Any)` | key: The key. value: The value. | ``True`` when the key was new, ``False`` when it was replaced. | O(1) expected (O(len(key)) to hash) | O(1) amortised |
| `HashSetTable.get(key: str, default: Any = None)` | key: The key to find. default: Returned when the key is absent. | The stored value, or ``default``. | O(1 + chain length) expected | O(1) |
| `HashSetTable.getitem(key: str)` | key: The key to find. | The stored value. | O(1 + chain length) expected | O(1) |
| `HashSetTable.contains(key: str)` | key: The key to test. | ``True`` when present. | O(1 + chain length) expected | O(1) |
| `HashSetTable.delete(key: str)` | key: The key to remove. | ``True`` when an entry was removed. | O(1 + chain length) expected | O(1) |
| `HashSetTable.keys()` | - | Iterator[str] | O(n + capacity) | O(1) beyond the generator frame |
| `HashSetTable.values()` | - | Iterator[Any] | O(n + capacity) | O(1) |
| `HashSetTable.items()` | - | Iterator[tuple[str, Any]] | O(n + capacity) | O(1) |
| `HashSetTable.bulk_put(pairs: Iterable[tuple[str, Any]])` | pairs: Iterable of ``(key, value)``. | None | O(m) expected for m pairs (plus one possible resize) | O(m) |
| `HashSetTable.load_factor()` | - | float | O(1) | O(1) |
| `HashSetTable.chain_collisions()` | - | The count of entries sharing a bucket with another entry. | O(capacity) | O(1) |
| `HashSetTable.capacity()` | - | int | O(1) | O(1) |
| `HashSetTable.copy()` | - | A new table with the same key/value pairs and its own buckets. | O(n + capacity) | O(n + capacity) |
| `HashSetTable.equals(other: object)` | other: Another :class:`HashSetTable`, or an object exposing ``items()``. | ``True`` when the mappings agree. | O(n + capacity) per table | O(1) |
| `PostingsTable.add_posting(term: str, doc_id: int, frequency: int = 1)` | term: Vocabulary term. doc_id: Document identifier. frequency: Term frequency inside that document. | None | O(1) expected for the table probe + amortised O(1) append | O(1) amortised |
| `PostingsTable.get_postings(term: str)` | term: Vocabulary term. | List of ``(doc_id, tf)`` pairs. | O(1) expected | O(1) (no copy) |
| `PostingsTable.document_frequency(term: str)` | term: Vocabulary term. | The length of the postings list, or 0. | O(1) expected | O(1) |

## structures/min_heap.py -- binary heap and top-K

| function | input | output | time | space |
|---|---|---|---|---|
| `MinHeap.is_empty()` | - | bool | O(1) | O(1) |
| `MinHeap.push(item: RankedItem)` | item: ``(score, doc_id)``. | None | O(log n) | O(1) |
| `MinHeap.pop()` | - | The root item, or ``None`` when empty. | O(log n) | O(1) |
| `MinHeap.peek()` | - | The root item, or ``None`` when empty. | O(1) | O(1) |
| `MinHeap.items()` | - | A shallow copy. | O(n) | O(n) |
| `TopKHeap.push(score: float, doc_id: int)` | score: Relevance score. doc_id: Document identifier used for deterministic tie-breaking. | ``True`` when the candidate was retained. | O(log k) -- O(1) when the heap is full and the candidate loses | O(1) |
| `TopKHeap.get_top_k()` | - | Up to ``k`` ``(score, doc_id)`` pairs, best first. | O(k log k) | O(k) |
| `sort_top_k(candidates: List[RankedItem], k: int)` | candidates: ``(score, doc_id)`` pairs. k: How many to keep. | The k best pairs, best first. | O(n log n) | O(n) |

## structures/graph.py -- citation graph and BFS

| function | input | output | time | space |
|---|---|---|---|---|
| `CitationGraph.add_node(node: str)` | node: Node identifier. | None | O(1) expected | O(1) |
| `CitationGraph.add_edge(source: str, target: str, edge_type: str = "cited")` | source: Citing case. target: Cited case. edge_type: Citation class (``cited``, ``applied``, ...). | None | O(1) expected (plus O(degree) duplicate check) | O(1) |
| `CitationGraph.neighbors(node: str, direction: Direction = "out", edge_type: Optional[str] = None)` | node: Source node. direction: ``'out'`` (cites), ``'in'`` (cited by) or ``'both'``. edge_type: When given, keep only edges of this class. | ``(neighbour, edge_type)`` pairs, deduplicated by ``(neighbour, edge_type)`` and stable in insertion order. | O(deg(node)) | O(deg(node)) |
| `CitationGraph.has_node(node: str)` | node: Node identifier. | ``True`` when the node is known. | O(1) expected | O(1) |
| `CitationGraph.nodes()` | - | Iterator[str] | O(V) | O(1) |
| `CitationGraph.node_count()` | - | int | O(1) | O(1) |
| `CitationGraph.edge_count()` | - | int | O(1) | O(1) |
| `CitationGraph.average_degree()` | - | ``E / V``, or 0 for an empty graph. | O(1) | O(1) |
| `CitationGraph.edge_type_counts()` | - | ``(edge_type, count)`` pairs sorted by count desc then name asc. | O(T log T) for T distinct classes | O(T) |
| `CitationGraph.degree(node: str, direction: Direction = "out")` | node: The node. direction: ``'out'``, ``'in'`` or ``'both'``. | Deduplicated degree. | O(deg(node)) | O(deg(node)) |
| `CitationGraph.bfs_traverse(start: str, max_depth: int = 2, edge_type: Optional[str] = None, direction: Direction = "out")` | start: Seed node. max_depth: Maximum number of hops from ``start`` (0 = start only). edge_type: Optional citation class filter. direction: ``'out'``, ``'in'`` or ``'both'``. | Tuples ``(node, depth, parent, edge_type)`` in BFS order; the seed has depth 0 and ``parent=None``. Each node appears once, at its minimum depth (the defining property of BFS). | O(V + E) bounded by the horizon: each node is enqueued once and each edge of a discovered node scanned once | O(V) for the visited table plus the queue |
| `CitationGraph.shortest_path(source: str, target: str)` | source: Start node. target: End node. | Node list including both endpoints, or ``None`` when unreachable. | O(V + E) | O(V) |
| `CitationGraph.connected_component_size(start: str, direction: Direction = "both")` | start: Seed node. direction: ``'both'`` for weak connectivity, ``'out'`` for reachability, ``'in'`` for reverse reachability. | Number of reachable nodes including ``start``. | O(V + E) | O(V) |
| `CitationGraph.depth_profile(start: str, max_depth: int = 3, direction: Direction = "both", edge_type: Optional[str] = None)` | start: Seed node. max_depth: Horizon. direction: Edge direction. edge_type: Optional class filter. | ``(depth, cumulative_reached)`` pairs including depth 0. | O(V + E) | O(V) |

## structures/sparse_vector.py -- sparse vectors

| function | input | output | time | space |
|---|---|---|---|---|
| `SparseVector.from_pairs(cls, dimension: int, pairs: Iterable[Tuple[int, float]])` | dimension: Logical dimensionality. pairs: Iterable of ``(index, value)``; zero values are dropped. | A :class:`SparseVector`. | O(m log m) for m pairs | O(m) |
| `SparseVector.zeros(cls, dimension: int)` | dimension: Logical dimensionality. | An empty sparse vector. | O(1) | O(1) |
| `SparseVector.nnz()` | - | int | O(1) | O(1) |
| `SparseVector.norm()` | - | ``sqrt(sum(v_i^2))``. | O(nnz) on first call, O(1) afterwards | O(1) |
| `SparseVector.norm_sq()` | - | ``sum(v_i^2)``. | O(nnz) | O(1) |
| `SparseVector.dot(other: "SparseVector")` | other: The other sparse vector (same logical dimension). | ``sum_i a_i * b_i``. | O(nnz_a + nnz_b) via a merge walk | O(1) |
| `SparseVector.dot_dense(dense: Sequence[float])` | dense: Dense vector of length ``self.dimension``. | ``sum_i v_i * dense[i]``. | O(nnz) | O(1) |
| `SparseVector.cosine(other: "SparseVector")` | other: The other sparse vector. | Value in ``[-1, 1]``; 0 when either vector is zero. | O(nnz_a + nnz_b) | O(1) |
| `SparseVector.add(other: "SparseVector")` | other: The other sparse vector. | ``self + other``. | O(nnz_a + nnz_b) | O(nnz_a + nnz_b) |
| `SparseVector.scale(factor: float)` | factor: Scalar multiplier. | A new sparse vector. | O(nnz) | O(nnz) |
| `SparseVector.squared_distance(other: "SparseVector")` | other: The other sparse vector. | ``\|\|self - other\|\|^2``. | O(nnz_a + nnz_b) | O(1) |
| `SparseVector.to_pairs()` | - | List[Tuple[int, float]] | O(nnz) | O(nnz) |

## core/tokenizer.py -- tokenisation and stemming

| function | input | output | time | space |
|---|---|---|---|---|
| `is_stop_word(token: str)` | token: A lower-case token. | ``True`` when the token is a stop word. | O(log S) for S = ``len(STOP_WORDS)`` (a C-level binary search) | O(1) |
| `tokenize(text: str, min_length: int = MIN_TOKEN_LENGTH, remove_stop_words: bool = True, stem: bool = False)` | text: Raw text. min_length: Tokens shorter than this are dropped. remove_stop_words: Drop entries of :data:`STOP_WORDS`. stem: Apply :func:`stem` to every surviving token. | Tokens in original order, lower-cased, with ``'`` dropped and digits kept (``"r2d2"``, ``"2006"``). | O(len(text)) -- one character scan plus O(1) work per token (and O(len(token)) when stemming) | O(number of tokens) |
| `stem(text: str)` | text: A single lower-case token. | The stemmed token. | O(len(text)) | O(len(text)) |
| `stem_token(token: str)` | token: A lower-case token. | The stem. Tokens of 3 characters or fewer, known root words such as ``"news"``, and non-alphabetic tokens are returned unchanged. | O(len(token)) per rule with a constant number of rules | O(len(token)) |
| `Tokenizer.tokenize(text: str)` | text: Raw text. | Tokens. | O(len(text)) | O(number of tokens) |
| `term_frequencies(tokens: List[str])` | tokens: Token list. | ``(term, count)`` pairs ordered by first appearance, which makes index construction deterministic and diff-friendly. | O(n log n) for the sort plus O(u) for the runs | O(n) for the sorted copy |

## core/inverted_index.py -- inverted index

| function | input | output | time | space |
|---|---|---|---|---|
| `CandidateSet.empty(cls, size: int)` | size: Corpus size. | An empty :class:`CandidateSet`. | O(N) | O(N) bytes |
| `CandidateSet.mark(doc_id: int)` | doc_id: Document identifier. | ``True`` when it was newly flagged. | O(1) | O(1) |
| `CandidateSet.contains(doc_id: int)` | doc_id: Document identifier. | ``True`` when the id is a candidate. | O(1) | O(1) |
| `CandidateSet.ids()` | - | Ascending ids. | O(N) | O(c) |
| `InvertedIndex.add_document(document: Document)` | document: The document to add; its ``doc_id`` must equal ``len(self.doc_lengths)`` so that arrays stay dense. | None | O(len(text)) plus O(u) postings appends for u unique terms | O(u) |
| `InvertedIndex.build(cls, documents: Sequence[Document], tokenizer: Optional[Tokenizer] = None, name: str = "")` | documents: Documents with dense ids ``0..n-1``. tokenizer: Tokenizer to use. name: Index name. | The populated index. | O(total tokens) | O(total unique postings) |
| `InvertedIndex.N()` | - | int | O(1) | O(1) |
| `InvertedIndex.average_document_length()` | - | ``total_length / N``, or 0 for an empty index. | O(1) | O(1) |
| `InvertedIndex.document_frequency(term: str)` | term: Vocabulary term. | Document frequency. | O(1) expected | O(1) |
| `InvertedIndex.postings_for(term: str)` | term: Vocabulary term. | ``[(doc_id, tf), ...]`` (the live list, not a copy). | O(1) expected | O(1) |
| `InvertedIndex.vocabulary_size()` | - | int | O(1) | O(1) |
| `InvertedIndex.terms()` | - | Iterator[str] | O(T + capacity) | O(1) |
| `InvertedIndex.collection_statistics()` | - | Mapping with ``N``, ``avgdl``, ``vocabulary``, ``tokens``, ``postings``, ``load_factor``, ``collisions`` (lifetime insert collisions, including rehashes) and ``chain_collisions`` (entries sharing a bucket in the final table). | O(T + capacity) to count postings and scan buckets | O(1) |
| `InvertedIndex.restrict(candidates: object, pool: Optional[CandidateSet] = None)` | candidates: Anything exposing ``contains(doc_id) -> bool`` -- a :class:`CandidateSet` or a :class:`ClusterFilter`. pool: Optional empty bitmap the scorer fills with the ids it actually scored, so the final top-K pass touches only those instead of the whole corpus. Defaults to an empty bitmap of corpus size. | A read-only view that :class:`core.bm25.BM25` can score. | O(N) to allocate the pool bitmap | O(N) bytes |
| `InvertedIndex.to_state()` | - | State mapping. Title, snippet, external id, label and source are stored verbatim; postings are stored as a list of ``(term, postings)`` pairs because the custom table is not itself picklable without exposing internals. | O(T + postings) | O(T + postings) |
| `InvertedIndex.from_state(cls, state: dict)` | state: Output of :meth:`to_state`. | A populated index. | O(T + postings) | O(T + postings) |
| `ClusterFilter.contains(doc_id: int)` | doc_id: Document identifier. | ``True`` when the document is a candidate. | O(1) | O(1) |
| `ClusterFilter.pool_size()` | - | Candidate pool size. | O(N) | O(1) |
| `RestrictedView.N()` | - | int | O(1) | O(1) |
| `RestrictedView.average_document_length()` | - | float | O(1) | O(1) |
| `RestrictedView.doc_lengths()` | - | List[int] | O(1) | O(1) |
| `RestrictedView.tokenizer()` | - | Tokenizer | O(1) | O(1) |
| `RestrictedView.document_frequency(term: str)` | term: Vocabulary term. | Global document frequency. | O(1) expected | O(1) |
| `RestrictedView.postings_for(term: str)` | term: Vocabulary term. | ``[(doc_id, tf), ...]``. | O(1) | O(1) |
| `RestrictedView.collection_statistics()` | - | Statistics with ``N`` replaced by the candidate count. | O(T) | O(1) |

## core/bm25.py -- BM25 scoring

| function | input | output | time | space |
|---|---|---|---|---|
| `BM25.idf(document_frequency: int, n_docs: int)` | document_frequency: ``n_t`` -- number of documents containing the term. n_docs: ``N`` -- corpus size. | ``ln(1 + (N - n + 0.5) / (n + 0.5))``; 0 when the term is unknown or the corpus is empty. The ``+1`` inside the logarithm keeps the numerator above the denominator, so idf stays strictly positive even for a term present in every document -- this is the standard Lucene variant and is what makes single-term corpora scoreable. | O(1) | O(1) |
| `BM25.term_score(term_frequency: int, document_length: int, average_length: float, idf: float)` | term_frequency: ``tf`` in the document. document_length: ``dl`` in tokens. average_length: ``avgdl`` of the corpus. idf: Precomputed idf for the term. | The BM25 term contribution. | O(1) | O(1) |
| `BM25.search(query: str, k: int = 10)` | query: Raw query string. k: Number of results. | ``(score, doc_id)`` pairs, best first. | O(sum_t \|postings(t)\| + P log K) where P is the number of postings actually scored -- ``O(P log K)`` overall | O(D + K) for the accumulation array (D = corpus size) and the heap |
| `BM25.score_document(query: str, doc_id: int)` | query: Raw query string. doc_id: Document identifier. | The score (0 when the document shares no term with the query). | O(number of query terms x cost of a posting lookup) -- for this index representation that is O(T_q * postings(t)) worst case | O(1) |
| `BM25.explain(query: str, doc_id: int)` | query: Raw query string. doc_id: Document identifier. | ``(term, tf, idf, contribution)`` rows in query-term order, with zero-contribution rows included so the trace is complete. | O(sum_t \|postings(t)\|) | O(number of query terms) |
| `BM25.explain_text(query: str, doc_id: int)` | query: Raw query string. doc_id: Document identifier. | A multi-line report with the formula and each term's numbers. | O(sum_t \|postings(t)\|) | O(number of query terms) |
| `build_scorer(index: InvertedIndex)` | index: A built inverted index. | A scorer bound to the full index. | O(1) | O(1) |

## ml/vectorizer.py -- TF-IDF vectoriser

| function | input | output | time | space |
|---|---|---|---|---|
| `idf_smoothed(document_frequency: int, n_documents: int)` | document_frequency: Number of documents containing the term. n_documents: Corpus size. | ``ln((1 + N) / (1 + df)) + 1``; always >= 1 so every term keeps weight and an unseen term cannot produce an infinite weight. | O(1) | O(1) |
| `TFIDFVectorizer.fit(documents: Sequence[Document])` | documents: Corpus to fit on. | ``self``, fitted. | O(total tokens + V log V) for vocabulary ranking | O(V + M) |
| `TFIDFVectorizer.dimension()` | - | int | O(1) | O(1) |
| `TFIDFVectorizer.transform_text(text: str)` | text: Raw text. | A unit-length sparse vector of length :attr:`dimension` (or the zero vector when no term is in the vocabulary). | O(len(text) + u log u) for u unique in-vocabulary terms (the log comes from sorting the pairs) | O(u) |
| `TFIDFVectorizer.transform(documents: Sequence[Document])` | documents: Documents already used at fit time (or new ones). | One vector per document, in order. | O(sum of tokenisation costs) | O(total nnz) |
| `TFIDFVectorizer.to_state()` | - | A picklable mapping. ``vocabulary`` is flattened to a list of ``(term, column)`` pairs so the cache payload carries no custom class besides the ones ``engine.py`` documents; ``from_state`` rebuilds the :class:`HashSetTable`. | O(M) | O(M) |
| `TFIDFVectorizer.from_state(cls, state: Dict[str, object])` | state: Output of :meth:`to_state`. | A fitted vectorizer. | O(M) | O(M) |

## ml/kmeans.py -- K-Means

| function | input | output | time | space |
|---|---|---|---|---|
| `KMeans.fit(vectors: Sequence[SparseVector])` | vectors: Training vectors, all of the same dimension. | ``self``, fitted. | O(n * k * nnz * iterations) using the sparse-dense distance identity | O(k * M + n) for centroids, accumulator and labels |
| `KMeans.predict(vector: SparseVector)` | vector: Query vector. | Cluster index. | O(k * nnz) | O(1) |
| `KMeans.predict_top_n(vector: SparseVector, n: int = 2)` | vector: Query vector. n: How many clusters to return. | ``(cluster_index, squared_distance)`` pairs. | O(k * nnz + k log k) | O(k) |
| `KMeans.predict_batch(vectors: Sequence[SparseVector])` | vectors: Query vectors. | Cluster index per vector. | O(n * k * nnz) | O(n) |
| `KMeans.cluster_members(labels: Sequence[int])` | labels: One label per document. | ``k`` lists of document indices. | O(n) | O(n) |
| `KMeans.to_state()` | - | A picklable mapping. | O(k * M) | O(k * M) |
| `KMeans.from_state(cls, state: Dict[str, object])` | state: Output of :meth:`to_state`. | A fitted :class:`KMeans`. | O(k * M) | O(k * M) |

## ml/evaluation.py -- clustering metrics

| function | input | output | time | space |
|---|---|---|---|---|
| `contingency_table(labels_true: Sequence[str], labels_pred: Sequence[int])` | labels_true: Ground-truth labels, one per document. labels_pred: Predicted cluster ids, one per document. | Sparse contingency table ``(true, pred) -> count`` held in a :class:`~structures.hash_table.HashSetTable`, so the metrics run on the project's own structure rather than a built-in ``dict``. | O(n) | O(c * j) for c true classes and j clusters actually co-occurring |
| `purity(labels_true: Sequence[str], labels_pred: Sequence[int])` | labels_true: Ground-truth labels. labels_pred: Predicted cluster ids. | Value in ``[0, 1]``; 0 for an empty corpus. | O(n) | O(c * j) for the contingency table |
| `mutual_information(labels_true: Sequence[str], labels_pred: Sequence[int])` | labels_true: Ground-truth labels. labels_pred: Predicted cluster ids. | Non-negative mutual information. | O(n + c * j) | O(c * j) |
| `nmi(labels_true: Sequence[str], labels_pred: Sequence[int])` | labels_true: Ground-truth labels. labels_pred: Predicted cluster ids. | Value in ``[0, 1]``; 1.0 when both partitions are single-class (the degenerate case sklearn also reports as 1.0); 0 for an empty corpus. | O(n + c * j) | O(c * j) |
| `adjusted_rand_index(labels_true: Sequence[str], labels_pred: Sequence[int])` | labels_true: Ground-truth labels. labels_pred: Predicted cluster ids. | Value in ``[-0.5, 1]``; 1.0 for identical partitions (including the degenerate all-in-one-cluster case); 0.0 for an empty corpus. | O(n + c * j) | O(c * j) |
| `evaluate_clustering(labels_true: Sequence[str], labels_pred: Sequence[int])` | labels_true: Ground-truth labels. labels_pred: Predicted cluster ids. | Mapping with ``purity``, ``nmi``, ``ari``, ``n``, ``n_true``, ``n_pred`` and ``largest_cluster_fraction``. | O(n + c * j) -- the contingency table is built four times by the individual metrics, which is still linear and keeps each function standalone and testable | O(c * j) |
| `confusion_matrix(labels_true: Sequence[str], labels_pred: Sequence[int])` | labels_true: Ground-truth labels. labels_pred: Predicted cluster ids. | ``(class_names, cluster_ids, matrix)`` where ``matrix[class][cluster]`` holds a count. Both axes are sorted for stable output. | O(n + c * j) | O(c * j) |

## core/agent.py -- the ReAct agent

| function | input | output | time | space |
|---|---|---|---|---|
| `AgentResult.total_ms()` | - | float | O(1) | O(1) |
| `AgentResult.to_dict()` | - | Mapping with ``intent``, ``path``, ``mode``, ``results``, ``logs``, ``timings``, ``stats``, ``total_ms`` and ``answer``. | O(number of results) | O(same) |
| `is_relational(query: str)` | query: Raw query. | ``True`` when a relational keyword appears as a word in the query. | O(len(query)) | O(1) |
| `ReActAgent.reason(query: str, mode: str)` | query: Raw query. mode: Requested mode (recorded in the log for context). | ``(intent, log_lines)``. | O(len(query)) | O(1) |
| `ReActAgent.act(query: str, intent: str, mode: str, k: int, probe_clusters: int, bfs_depth: int, edge_type: Optional[str] = None)` | query: Raw query. intent: Result of :meth:`reason`. mode: ``"baseline"`` or ``"ml"``. k: Number of results. probe_clusters: Clusters to probe on the ML path. bfs_depth: BFS horizon on the relational path. edge_type: Optional citation-class filter for the relational BFS (``"cited"``, ``"applied"``, ...). ``None`` follows every class. | ``(path, payload)`` where the payload carries ranked ids, stats and log lines. | O(P log K) on the two lexical paths and O(V + E) on the relational path, where P is the number of postings scored, V/E the graph size and K the requested result count | O(D + K) for the dense score array (D = corpus size) plus the result heap |
| `ReActAgent.observe(path: str, payload: Dict[str, object], k: int, baseline_ranked: Optional[List[Tuple[float, int]]] = None)` | path: The path taken. payload: Output of :meth:`act`. k: Number of results requested. baseline_ranked: Baseline ranking, when comparing paths. | ``(result_rows, log_lines)``. | O(k) | O(k) |
| `ReActAgent.answer(path: str, rows: Sequence[Dict[str, object]], query: str)` | path: The path taken. rows: Result rows. query: Original query. | One-paragraph answer; honest about an empty result set. | O(number of rows) | O(same) |
| `ReActAgent.run(query: str, mode: str = "baseline", k: int = 10, probe_clusters: int = DEFAULT_PROBE_CLUSTERS, bfs_depth: int = DEFAULT_BFS_DEPTH, compare_baseline: bool = False, edge_type: Optional[str] = None)` | query: Raw query. mode: ``"baseline"`` or ``"ml"``. k: Number of results. probe_clusters: Clusters to probe on the ML path. bfs_depth: BFS horizon on the relational path. compare_baseline: Also run the baseline so :meth:`observe` can report recall. Costs a second full search. edge_type: Optional citation-class filter for the relational BFS. | An :class:`AgentResult`. | dominated by the chosen path: O(P log K) for lexical, O(V + E) for relational, times a factor of two when ``compare_baseline`` is set | O(D + K) |
