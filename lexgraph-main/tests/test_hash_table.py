"""Randomised and structural tests for the custom hash table."""

from __future__ import annotations

import math
import random

import pytest

from structures.hash_table import HashSetTable, PostingsTable, polynomial_hash


def test_polynomial_hash_is_deterministic_and_bounded():
    for key in ("", "a", "hello", "texas", "the quick brown fox", "x" * 500):
        value = polynomial_hash(key)
        assert value == polynomial_hash(key)
        assert 0 <= value < 2**61 - 1


def test_polynomial_hash_matches_horner_evaluation():
    key = "abcdef"
    expected = 0
    for char in key:  # left-to-right Horner evaluation
        expected = (expected * 31 + ord(char)) % (2**61 - 1)
    assert polynomial_hash(key) == expected
    positional = sum(ord(c) * 31 ** (len(key) - 1 - i) for i, c in enumerate(key))
    assert polynomial_hash(key) == positional % (2**61 - 1)


def test_polynomial_hash_distinguishes_anagrams():
    pairs = [("listen", "silent"), ("tea", "eat"), ("drawer", "reward"), ("cite", "etic")]
    for a, b in pairs:
        assert polynomial_hash(a) != polynomial_hash(b)


def test_put_get_contains_delete():
    table = HashSetTable()
    assert table.put("alpha", 1) is True
    assert table.put("beta", 2) is True
    assert table.put("alpha", 3) is False
    assert table.get("alpha") == 3
    assert table.get("beta") == 2
    assert table.get("gamma") is None
    assert table.get("gamma", "fallback") == "fallback"
    assert table.contains("beta") is True
    assert "beta" in table
    assert table.delete("beta") is True
    assert table.delete("beta") is False
    assert table.contains("beta") is False
    assert len(table) == 1


def test_resize_preserves_entries_and_keeps_load_factor_bounded():
    table = HashSetTable(capacity=4)
    keys = [f"key-{i}" for i in range(500)]
    for index, key in enumerate(keys):
        table.put(key, index)
        assert table.load_factor <= 0.75 + 1e-12
    for index, key in enumerate(keys):
        assert table.get(key) == index
    assert len(table) == 500
    assert table.capacity >= 500


def test_property_against_builtin_dict():
    """Insert/overwrite/delete a random workload and match ``dict`` exactly."""
    rng = random.Random(2024)
    table = HashSetTable()
    reference: dict[str, int] = {}
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    for step in range(4000):
        action = rng.random()
        key = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        if action < 0.55:
            value = rng.randint(-10**6, 10**6)
            table.put(key, value)
            reference[key] = value
        elif action < 0.75:
            assert table.delete(key) is (key in reference)
            reference.pop(key, None)
        else:
            assert table.get(key) == reference.get(key)
        if step % 200 == 0:
            assert len(table) == len(reference)


def test_keys_and_items_match_dict_after_churn():
    rng = random.Random(99)
    table = HashSetTable()
    reference: dict[str, int] = {}
    for _ in range(1500):
        key = "w" + str(rng.randint(0, 400))
        if rng.random() < 0.7:
            value = rng.randint(0, 99)
            table.put(key, value)
            reference[key] = value
        else:
            table.delete(key)
            reference.pop(key, None)
    assert sorted(table.keys()) == sorted(reference)
    assert dict(table.items()) == reference
    assert sorted(table.values()) == sorted(reference.values())


def test_keyerror_on_missing_subscript():
    table = HashSetTable()
    table["present"] = 5
    assert table["present"] == 5
    with pytest.raises(KeyError):
        _ = table["absent"]


def test_bulk_put_sizes_once():
    table = HashSetTable()
    table.bulk_put((f"term{i}", i) for i in range(1000))
    assert len(table) == 1000
    assert table.capacity >= 1000
    assert table.get("term999") == 999


def test_collision_counter_is_recorded():
    table = HashSetTable(capacity=4)
    # With only 4 buckets these keys are guaranteed to collide.
    for index in range(20):
        table.put(f"k{index}", index)
    assert table.collisions > 0
    assert table.load_factor <= 0.75 + 1e-12


def test_chain_collisions_matches_uniform_prediction():
    """`chain_collisions` must follow `n - C(1 - e^(-n/C))` for random keys.

    This is the metric the report compares against the theory, so it is pinned
    here: a hash that started clustering keys would break the ratio.
    """
    rng = random.Random(11)
    keys = [f"term{rng.randrange(10**8)}x{index}" for index in range(20_000)]
    table = HashSetTable(capacity=65_536)  # alpha ~ 0.31, no resize during load
    table.bulk_put((key, None) for key in keys)

    n, capacity = len(keys), table.capacity
    predicted = n - capacity * (1.0 - math.exp(-n / capacity))
    ratio = table.chain_collisions / predicted
    assert 0.90 <= ratio <= 1.15, (
        f"chain collisions {table.chain_collisions} vs predicted {predicted:.0f}"
    )


def test_lifetime_counter_exceeds_final_table_collisions_after_resizes():
    """The lifetime counter counts rehash insertions; the final one does not."""
    keys = [f"key{index}" for index in range(5_000)]
    grown = HashSetTable(capacity=16)  # many resizes -> rehash re-insertions
    for key in keys:
        grown.put(key, None)
    one_shot = HashSetTable(capacity=16)
    one_shot.bulk_put((key, None) for key in keys)

    assert grown.capacity == one_shot.capacity
    assert grown.chain_collisions == one_shot.chain_collisions
    assert grown.collisions > one_shot.collisions


def test_chain_collisions_is_zero_without_collisions():
    table = HashSetTable(capacity=1024)
    for index in range(50):
        table.put(f"k{index}", index)
    assert table.chain_collisions == 0


def test_postings_table_adds_and_reads():
    postings = PostingsTable()
    postings.add_posting("graphics", 1, 3)
    postings.add_posting("graphics", 4, 1)
    postings.add_posting("kernel", 2)
    assert postings.get_postings("graphics") == [(1, 3), (4, 1)]
    assert postings.get_postings("missing") == []
    assert postings.document_frequency("graphics") == 2
    assert postings.document_frequency("missing") == 0
    assert len(postings) == 2
