"""Custom hash table: polynomial rolling hash, separate chaining, dynamic resize.

This is the structure underneath the inverted index and the adjacency list, so
it is deliberately written to support *mutation* (``put`` overwrite, ``delete``
with tombstone-free chain splicing) as well as bulk ``keys`` iteration.

Nothing in this module uses a built-in ``dict`` or ``set``.

Design notes
------------
* **Polynomial rolling hash**, base 31, modulo ``2**61 - 1``.  Evaluated with
  Horner's rule so each character costs one multiply-add:
  ``h = (h * 31 + ord(c)) mod p``.
* **Separate chaining** with an array of bucket heads and a hand-written
  singly linked list of nodes.  Chaining (rather than open addressing) keeps
  deletion simple and keeps performance stable under a clumpy hash, which
  matters because our keys are English words, not random bytes.
* **Power-of-two capacity** so the modulo can be replaced by mask arithmetic.
* **Load factor 0.75** triggers a capacity doubling plus a full rehash.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional

from config import (
    HASH_BASE,
    HASH_INITIAL_CAPACITY,
    HASH_LOAD_FACTOR_MAX,
    HASH_MODULUS,
)


_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    """SplitMix64 finaliser, used to spread integer keys.

    Args:
        value: Any integer (negative or positive).

    Returns:
        A well-mixed 64-bit integer.

    Complexity:
        Time O(1); space O(1).
    """
    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


class _Entry:
    """One chained node.

    Attributes:
        key: The stored key.
        value: The stored value.
        next: The next node in this bucket's chain, or ``None``.

    Space: O(1) per entry.
    """

    __slots__ = ("key", "value", "next")

    def __init__(self, key: Any, value: Any, next_entry: Optional["_Entry"]) -> None:
        """Create one chain node.

        Args:
            key: The stored key.
            value: The stored value.
            next_entry: The next node in this bucket's chain, or ``None``.

        Complexity: Time O(1); space O(1).
        """
        self.key = key
        self.value = value
        self.next = next_entry


def polynomial_hash(key: str, base: int = HASH_BASE, modulus: int = HASH_MODULUS) -> int:
    """Compute the polynomial rolling hash of ``key``.

    Uses Horner's rule: ``h_0 = 0``, ``h_{i+1} = (h_i * base + ord(c_i)) mod p``.
    The result is therefore the polynomial evaluation
    ``sum(ord(c_i) * base**(L-1-i)) mod p`` -- the leftmost character carries
    the highest weight, exactly like positional notation.

    Args:
        key: The string to hash.
        base: Multiplicative base (default 31).
        modulus: Prime modulus (default ``2**61 - 1``).

    Returns:
        A non-negative integer in ``[0, modulus)``.

    Complexity:
        Time O(L) where L = ``len(key)``; space O(1).

    Why base 31: for lower-case ASCII the distinct character values are
    consecutive, so ``31 * h`` separates anagrams far better than base 2 while
    staying small enough that ``h * base`` plus a codepoint fits in a single
    Python machine word for typical token lengths.
    """
    h = 0
    for ch in key:
        h = (h * base + ord(ch)) % modulus
    return h


def hash_key(key: Any) -> int:
    """Hash any key of the types the engine uses.

    Strings take the polynomial rolling hash above; integers are mixed through
    a SplitMix64 finaliser; tuples are folded element by element.  The integer
    branch matters because several engine tables are keyed by document id,
    cluster id or column id, and ``k mod p`` alone would leave the low bits of
    a small-range integer set unchanged.

    Args:
        key: A ``str``, an ``int`` or a (non-empty) tuple of those.

    Returns:
        A non-negative integer in ``[0, modulus)``.

    Raises:
        TypeError: For any other key type, or an empty tuple.

    Complexity:
        Time O(len(key)) for a string, O(1) for an integer, O(number of
        elements) for a tuple; space O(1).
    """
    if isinstance(key, str):
        return polynomial_hash(key)
    if isinstance(key, tuple):
        if not key:
            raise TypeError("empty tuple is not a valid key")
        folded = 0x243F6A8885A308D3
        for element in key:
            folded = _mix64(folded ^ hash_key(element))
        return folded
    if isinstance(key, bool) or not isinstance(key, int):
        raise TypeError(f"unsupported key type: {type(key).__name__}")
    return _mix64(key)


class HashSetTable:
    """A hash table mapping arbitrary hashable-besides-str keys to values.

    The public surface mirrors the subset of ``dict`` the engine needs:
    ``put``, ``get``, ``contains``, ``delete``, ``keys``, ``__len__``,
    ``__contains__``, ``__getitem__``/``__setitem__``.

    Attributes:
        _capacity: Number of buckets (always a power of two).
        _buckets: Array of chain heads.
        _size: Number of live entries.
        collisions: Cumulative count of *insert attempts* that found a
            non-empty bucket.  This includes the entries re-inserted by every
            resize, so it measures hashing work done during the table's whole
            lifetime, not the shape of the final table -- a table built by
            ``n`` ``put`` calls reports far more than the same table rebuilt
            through one :meth:`bulk_put`.  Use :attr:`chain_collisions` when
            the question is "how well did this key set distribute".
        probes: Total chain nodes visited across all lookups.

    Space: O(n + capacity) for n live entries.
    """

    __slots__ = ("_buckets", "_capacity", "_size", "collisions", "probes")

    def __init__(self, capacity: int = HASH_INITIAL_CAPACITY) -> None:
        """Create an empty table.

        Args:
            capacity: Initial bucket count; rounded up to a power of two.

        Complexity:
            Time O(capacity); space O(capacity).
        """
        cap = 1
        while cap < capacity:
            cap <<= 1
        self._capacity = cap
        self._buckets: List[Optional[_Entry]] = [None] * cap
        self._size = 0
        self.collisions = 0
        self.probes = 0

    # -- internals ---------------------------------------------------------

    def _bucket_index(self, key: str) -> int:
        """Map a key to a bucket index.

        Args:
            key: The key.

        Returns:
            An index in ``[0, capacity)``.

        Complexity:
            Time O(len(key)) (dominated by hashing); space O(1).

        Note:
            The ``str`` branch is inlined rather than delegated to
            :func:`hash_key`: string keys are the overwhelmingly common case
            (every term in every postings lookup), and one Python call layer
            per probe measured at ~9% of the index build.
        """
        mask = self._capacity - 1
        if type(key) is str:
            return polynomial_hash(key) & mask
        return hash_key(key) & mask

    def _resize(self, new_capacity: int) -> None:
        """Grow the bucket array and rehash every entry.

        Args:
            new_capacity: Requested capacity (rounded up to a power of two).

        Complexity:
            Time O(n + capacity) amortised over the insert that triggered it;
            space O(capacity).
        """
        cap = 1
        while cap < new_capacity:
            cap <<= 1
        old_buckets = self._buckets
        self._capacity = cap
        self._buckets = [None] * cap
        self._size = 0
        for head in old_buckets:
            node = head
            while node is not None:
                nxt = node.next
                self._insert_node(node.key, node.value)
                node = nxt

    def _insert_node(self, key: str, value: Any) -> bool:
        """Insert or overwrite without load-factor checks.

        Args:
            key: Key to write.
            value: Value to store.

        Returns:
            ``True`` if a new entry was created, ``False`` if one was updated.

        Complexity:
            Time O(1 + chain length) expected; space O(1).
        """
        index = self._bucket_index(key)
        head = self._buckets[index]
        if head is None:
            self._buckets[index] = _Entry(key, value, None)
            self._size += 1
            return True
        self.collisions += 1  # lifetime count, includes rehash insertions
        node = head
        while True:
            if node.key == key:
                node.value = value
                return False
            if node.next is None:
                break
            node = node.next
        node.next = _Entry(key, value, None)
        self._size += 1
        return True

    # -- public API --------------------------------------------------------

    def put(self, key: str, value: Any) -> bool:
        """Insert ``key`` mapping to ``value``, overwriting an existing key.

        Args:
            key: The key.
            value: The value.

        Returns:
            ``True`` when the key was new, ``False`` when it was replaced.

        Complexity:
            Time O(1) expected (O(len(key)) to hash); O(n) worst case when
            every key collides into one chain.  Space O(1) amortised.
        """
        created = self._insert_node(key, value)
        if self._size > self._capacity * HASH_LOAD_FACTOR_MAX:
            self._resize(self._capacity << 1)
        return created

    def __setitem__(self, key: str, value: Any) -> None:
        """Alias for :meth:`put` that matches ``dict`` semantics.

        Complexity: O(1) expected.
        """
        self.put(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Look up ``key``.

        Args:
            key: The key to find.
            default: Returned when the key is absent.

        Returns:
            The stored value, or ``default``.

        Complexity:
            Time O(1 + chain length) expected; space O(1).
        """
        node = self._buckets[self._bucket_index(key)]
        steps = 0
        while node is not None:
            steps += 1
            if node.key == key:
                self.probes += steps
                return node.value
            node = node.next
        self.probes += steps
        return default

    def getitem(self, key: str) -> Any:
        """Look up ``key`` and raise ``KeyError`` when it is absent.

        Args:
            key: The key to find.

        Returns:
            The stored value.

        Raises:
            KeyError: If the key is not present.

        Complexity:
            Time O(1 + chain length) expected; space O(1).
        """
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __getitem__(self, key: str) -> Any:
        """Subscript read; see :meth:`getitem`.

        Complexity: O(1) expected.
        """
        return self.getitem(key)

    def contains(self, key: str) -> bool:
        """Test membership without materialising the value.

        Args:
            key: The key to test.

        Returns:
            ``True`` when present.

        Complexity:
            Time O(1 + chain length) expected; space O(1).
        """
        node = self._buckets[self._bucket_index(key)]
        while node is not None:
            if node.key == key:
                return True
            node = node.next
        return False

    __contains__ = contains

    def delete(self, key: str) -> bool:
        """Remove ``key`` from the table.

        Args:
            key: The key to remove.

        Returns:
            ``True`` when an entry was removed.

        Complexity:
            Time O(1 + chain length) expected; space O(1).

        Note:
            The table does *not* shrink: a university-scale corpus builds the
            index once, so paying for hysteresis-free shrinking would only add
            churn.  Callers that need to reclaim memory should build a fresh
            table.
        """
        index = self._bucket_index(key)
        node = self._buckets[index]
        prev: Optional[_Entry] = None
        while node is not None:
            if node.key == key:
                if prev is None:
                    self._buckets[index] = node.next
                else:
                    prev.next = node.next
                self._size -= 1
                return True
            prev = node
            node = node.next
        return False

    def keys(self) -> Iterator[str]:
        """Yield every live key.

        Yields:
            Keys in bucket order.

        Complexity:
            Time O(n + capacity); space O(1) beyond the generator frame.
        """
        for head in self._buckets:
            node = head
            while node is not None:
                yield node.key
                node = node.next

    def values(self) -> Iterator[Any]:
        """Yield every live value.

        Yields:
            Values in bucket order.

        Complexity:
            Time O(n + capacity); space O(1).
        """
        for head in self._buckets:
            node = head
            while node is not None:
                yield node.value
                node = node.next

    def items(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(key, value)`` pairs.

        Yields:
            Pairs in bucket order.

        Complexity:
            Time O(n + capacity); space O(1).
        """
        for head in self._buckets:
            node = head
            while node is not None:
                yield node.key, node.value
                node = node.next

    def bulk_put(self, pairs: Iterable[tuple[str, Any]]) -> None:
        """Insert many pairs, sizing the table once up front.

        Args:
            pairs: Iterable of ``(key, value)``.

        Complexity:
            Time O(m) expected for m pairs (plus one possible resize);
            space O(m).
        """
        materialised = list(pairs)
        needed = 1
        while needed < (self._size + len(materialised)) / HASH_LOAD_FACTOR_MAX:
            needed <<= 1
        if needed > self._capacity:
            self._resize(needed)
        for key, value in materialised:
            self._insert_node(key, value)

    @property
    def load_factor(self) -> float:
        """Current ``size / capacity``.

        Complexity:
            Time O(1); space O(1).
        """
        return self._size / self._capacity

    @property
    def chain_collisions(self) -> int:
        """Keys sitting behind another key in the *current* table.

        Computed as ``size - number_of_non_empty_buckets``, i.e. the number of
        entries whose bucket probe had to walk past at least one other entry.
        Unlike :attr:`collisions` this is a property of the final table only,
        so it is the number to compare against the uniform-hashing prediction
        ``n - C(1 - e^(-n/C))``.

        Returns:
            The count of entries sharing a bucket with another entry.

        Complexity:
            Time O(capacity); space O(1).

        Note:
            This is a real traversal of the bucket array, not a counter, so it
            is meant for reporting rather than for the hot path.
        """
        non_empty = 0
        for head in self._buckets:
            if head is not None:
                non_empty += 1
        return self._size - non_empty

    @property
    def capacity(self) -> int:
        """Number of buckets currently allocated.

        Complexity: O(1).
        """
        return self._capacity

    def __len__(self) -> int:
        """Number of live entries.

        Complexity: O(1).
        """
        return self._size

    def copy(self) -> "HashSetTable":
        """Shallow copy of the table.

        Returns:
            A new table with the same key/value pairs and its own buckets.

        Complexity:
            Time O(n + capacity); space O(n + capacity).
        """
        clone: HashSetTable = self.__class__(self._capacity)
        clone.bulk_put(self.items())
        return clone

    def equals(self, other: object) -> bool:
        """Value equality against any table-like ``{key: value}`` mapping.

        Two tables are equal when they hold the same keys mapping to equal
        values, regardless of capacity or insertion order -- i.e. ``dict``
        semantics, so a round trip through :meth:`items` compares equal to the
        original.

        Args:
            other: Another :class:`HashSetTable`, or an object exposing
                ``items()``.

        Returns:
            ``True`` when the mappings agree.

        Complexity:
            Time O(n + capacity) per table; space O(1).
        """
        if other is self:
            return True
        try:
            other_items = list(other.items())  # type: ignore[union-attr]
        except AttributeError:
            return NotImplemented  # type: ignore[return-value]
        if len(other_items) != self._size:
            return False
        for key, value in other_items:
            sentinel = self._MISSING
            if self.get(key, sentinel) is sentinel:
                return False
            if self.get(key) != value:
                return False
        return True

    _MISSING = object()

    def __eq__(self, other: object) -> bool:
        """Equality with ``dict`` semantics; see :meth:`equals`.

        Complexity: O(n + capacity).
        """
        return self.equals(other)

    __hash__ = None  # type: ignore[assignment]

    def __iter__(self) -> Iterator[str]:
        """Iterate keys (alias for :meth:`keys`).

        Complexity: O(n + capacity).
        """
        return self.keys()

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return (
            f"HashSetTable(size={self._size}, capacity={self._capacity}, "
            f"load={self.load_factor:.2f})"
        )


class PostingsTable(HashSetTable):
    """Inverted-index convenience layer over :class:`HashSetTable`.

    Value layout: a *linked* postings list of ``(doc_id, term_frequency)``
    tuples.  A plain Python list is used for the chain because the engine
    appends during indexing and then iterates read-only at query time, so the
    list's amortised append and cache-friendly iteration beat a hand-rolled
    linked list here.

    The class additionally keeps ``document_frequency`` implicitly as
    ``len(postings)`` and exposes :meth:`get_postings` in the form the BM25
    scorer wants.
    """

    __slots__ = ()

    def add_posting(self, term: str, doc_id: int, frequency: int = 1) -> None:
        """Append one ``(doc_id, frequency)`` pair to ``term``'s postings.

        Args:
            term: Vocabulary term.
            doc_id: Document identifier.
            frequency: Term frequency inside that document.

        Complexity:
            Time O(1) expected for the table probe + amortised O(1) append;
            space O(1) amortised.
        """
        postings: Optional[list] = self.get(term)
        if postings is None:
            postings = []
            self.put(term, postings)
        postings.append((doc_id, frequency))

    def get_postings(self, term: str) -> list:
        """Return the postings list for ``term`` or an empty list.

        Args:
            term: Vocabulary term.

        Returns:
            List of ``(doc_id, tf)`` pairs.

        Complexity:
            Time O(1) expected; space O(1) (no copy).
        """
        postings = self.get(term)
        return postings if postings is not None else []

    def document_frequency(self, term: str) -> int:
        """Number of documents containing ``term``.

        Args:
            term: Vocabulary term.

        Returns:
            The length of the postings list, or 0.

        Complexity:
            Time O(1) expected; space O(1).
        """
        postings = self.get(term)
        return len(postings) if postings is not None else 0


__all__ = ["HashSetTable", "PostingsTable", "hash_key", "polynomial_hash"]
