"""Tokenizer and stemmer edge cases."""

from __future__ import annotations

import pytest

from core.tokenizer import (
    STOP_WORDS,
    Tokenizer,
    stem_token,
    term_frequencies,
    tokenize,
)


def test_basic_tokenisation_lowercases_and_splits():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_punctuation_and_apostrophes():
    assert tokenize("don't stop") == ["dont", "stop"]
    assert tokenize("it’s fine") == ["fine"]
    assert tokenize("e-mail: user@host") == ["mail", "user", "host"]


def test_stop_words_removed_and_kept_on_request():
    assert "the" not in tokenize("the cat sat on the mat")
    assert tokenize("the cat", remove_stop_words=False) == ["the", "cat"]


def test_min_length():
    assert tokenize("a b cd efg") == ["cd", "efg"]
    # min_length=1 keeps single-character words that are not stop words; "a"
    # and "b" are, so they stay filtered out.
    assert tokenize("a b cd efg", min_length=1) == ["b", "cd", "efg"]


def test_digits_and_alphanumerics_survive():
    assert tokenize("FCA [2006] 1234 r2d2") == ["fca", "2006", "1234", "r2d2"]


def test_empty_and_whitespace():
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []


def test_unicode_letters_are_not_tokens():
    # Non-ASCII letters are separators here: safe and predictable for this corpus.
    assert tokenize("café bar") == ["caf", "bar"]


@pytest.mark.parametrize(
    "word,expected",
    [
        ("running", "run"),
        ("rendering", "render"),
        ("writing", "write"),
        ("hopped", "hop"),
        ("studies", "study"),
        ("policies", "policy"),
        ("classes", "class"),
        ("news", "news"),
        ("references", "reference"),
        ("distinguished", "distinguish"),
        ("applied", "apply"),
        ("followed", "follow"),
        # Porter step 1a: "-sses" -> "-ss", independently of the stem's length.
        ("sses", "ss"),
        ("glasses", "glass"),
    ],
)
def test_stemmer(word, expected):
    assert stem_token(word) == expected


def test_stemmer_leaves_short_words_alone():
    for word in ("a", "be", "the", "cat", "go"):
        assert stem_token(word) == word


def test_stemmer_leaves_non_alpha_alone():
    assert stem_token("2006") == "2006"
    assert stem_token("r2d2") == "r2d2"


def test_stemmer_is_idempotent_for_test_vocabulary():
    for word in ("running", "rendering", "references", "distinguished"):
        once = stem_token(word)
        assert stem_token(once) == once


def test_tokenizer_object_matches_function():
    tokenizer = Tokenizer(min_length=3, remove_stop_words=True, use_stemmer=True)
    # "these" is a stop word, so it is dropped before the stemmer sees it.
    assert tokenizer("The rendering of these classes") == ["render", "class"]
    assert tokenizer.tokenize("The rendering") == ["render"]


def test_term_frequencies_preserve_first_seen_order():
    assert term_frequencies(["b", "a", "b", "c", "b"]) == [("b", 3), ("a", 1), ("c", 1)]
    assert term_frequencies([]) == []


def test_stop_word_table_has_no_duplicates():
    assert len(STOP_WORDS) == len({word for word in STOP_WORDS})
