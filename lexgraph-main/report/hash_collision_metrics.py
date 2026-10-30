"""Report the two hash-table collision metrics, measured three ways.

A bare "collisions" number is ambiguous for a table that resizes: every resize
re-inserts every entry, so a lifetime counter keeps counting the same keys. This
script records both definitions side by side so the report can quote the one it
means:

* ``chain_collisions`` -- entries sharing a bucket in the **final** table
  (``size - non-empty buckets``); the number the uniform-hashing prediction
  ``n - C(1 - e^(-n/C))`` should be compared against.
* ``collisions`` -- cumulative insert attempts that found a non-empty bucket,
  including rehash insertions; a measure of work done, not of table shape.

Writes ``report/tables/hash_collision_metrics.txt``.

Usage::

    python -m report.hash_collision_metrics
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

from config import TABLES_DIR


def uniform_prediction(n: int, capacity: int) -> float:
    """Expected number of entries sharing a bucket under uniform hashing.

    For ``n`` keys in ``C`` buckets the number of occupied buckets is
    ``C(1 - (1 - 1/C)^n) ≈ C(1 - e^(-n/C))``, so the expected number of entries
    that are not alone in their bucket is ``n`` minus that.

    Args:
        n: Number of keys.
        capacity: Number of buckets.

    Returns:
        The expectation, as a float.

    Complexity:
        Time O(1); space O(1).
    """
    return n - capacity * (1.0 - math.exp(-n / capacity))


def chain_lengths(table: object) -> List[int]:
    """Chain length of every bucket, including empty ones.

    Args:
        table: A ``HashSetTable``.

    Returns:
        One length per bucket.

    Complexity:
        Time O(n + capacity); space O(capacity).
    """
    lengths: List[int] = []
    for head in table._buckets:  # type: ignore[attr-defined]
        count = 0
        node = head
        while node is not None:
            count += 1
            node = node.next
        lengths.append(count)
    return lengths


def main() -> int:
    """Measure and write the comparison.

    Returns:
        Exit code.

    Complexity:
        Time O(total postings) for the rebuilds; space O(vocabulary).
    """
    from engine import get_engine
    from structures.hash_table import HashSetTable

    engine = get_engine(verbose=False)
    lines: List[str] = [
        "Hash-table collisions: lifetime counter vs final-table shape",
        "=" * 72,
        "",
        "A resize re-inserts every live entry, so a lifetime counter counts the",
        "same keys again. 'chain_collisions' is size - non-empty buckets in the",
        "final table and is the number to compare with uniform hashing.",
        "",
    ]
    rows: List[Tuple[str, int, int, int, float, int, int, int]] = []
    for name, index in (
        ("newsgroups", engine.news_index),
        ("legal", engine.legal_index),
    ):
        postings = index.postings
        size, capacity = postings._size, postings.capacity
        rebuilt = HashSetTable()
        for key in postings.keys():
            rebuilt.put(key, None)
        lengths = chain_lengths(rebuilt)
        final_collisions = rebuilt.chain_collisions
        prediction = uniform_prediction(size, capacity)
        rows.append(
            (
                name,
                size,
                capacity,
                int(final_collisions),
                size / capacity,
                int(prediction),
                rebuilt.collisions,
                postings.collisions,
            )
        )
        longest = max(lengths)
        mean_bucket = sum(lengths) / len(lengths)
        lines += [
            f"{name}: {size} distinct terms, {capacity} buckets, "
            f"load {size / capacity:.4f}",
            f"  final-table collisions (size - non-empty buckets) : {final_collisions}",
            f"  uniform-hashing prediction n - C(1 - e^(-n/C))    : {prediction:.0f}",
            f"  ratio measured / predicted                        : "
            f"{final_collisions / prediction:.3f}",
            f"  longest chain                                     : {longest}",
            f"  mean chain over ALL buckets                       : {mean_bucket:.4f}",
            f"  lifetime counter after a put()-by-put() build     : {rebuilt.collisions}",
            f"  lifetime counter after one bulk_put() load        : {postings.collisions}",
            "",
        ]

    path = os.path.join(str(TABLES_DIR), "hash_collision_metrics.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"[hash] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
