# BM25 / TF-IDF ranking comparison (identical tokenisation)

| implementation | query time | ratio vs mine | agreement |
|---|---|---|---|
| custom BM25 | 1.346 ms/query | 1.000 | reference |
| rank_bm25.BM25Okapi | 11.813 ms/query | 0.11x | top-10 overlap 99.6% |
| sklearn TF-IDF + cosine | 27.810 ms/query | 0.05x | top-10 overlap 65.2% |
