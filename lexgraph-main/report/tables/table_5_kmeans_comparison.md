# K-Means comparison on 4000 documents (k=20, same seed)

| implementation | fit time | inertia | NMI vs labels |
|---|---|---|---|
| custom K-Means | 11311 ms | 3789.9 | 0.3458 |
| sklearn.cluster.KMeans | 13334 ms | 3780.1 | 0.3539 |
