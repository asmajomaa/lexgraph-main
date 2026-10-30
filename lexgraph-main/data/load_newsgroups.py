"""20 Newsgroups loader (Mitchell, 1997).

Supports both shipped layouts:

* ``20news-bydate-{train,test}/<category>/<file>`` -- the raw "bydate" split.
* ``20news-18828/<category>/<file>`` -- the de-duplicated single split.

Documents are parsed for their ``From``/``Subject``/``Message-ID``/
``References``/``Newsgroup``/``Lines`` headers and the body is cleaned of
quoted replies (lines starting with ``>`` and ``...wrote:`` attributions) and
signatures (everything after the ``-- `` sig delimiter).

The folder name is the ground-truth label.

If no corpus is present on disk the loader falls back to
:func:`data.synthetic.generate_newsgroups_corpus` so tests and demos still run;
the returned :class:`NewsgroupsCorpus` records which source was used.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from config import MAX_DOCUMENT_CHARS
from data.documents import Document, first_nonempty, make_snippet

#: Header names we keep.  Parsing stops at the first blank line.
_WANTED_HEADERS = {
    "from",
    "subject",
    "message-id",
    "references",
    "newsgroup",
    "newsgroups",
    "lines",
    "organization",
    "date",
}

#: Marker that starts a signature block.
_SIG_DELIMITER = "-- "


class NewsgroupsCorpus:
    """A loaded 20 Newsgroups corpus plus the metadata the report needs.

    Attributes:
        documents: Indexable documents.
        categories: Sorted category names.
        source: ``"20news-bydate"``, ``"20news-18828"`` or ``"synthetic"``.
        root: Filesystem root that was read (``""`` for synthetic data).
    """

    __slots__ = ("documents", "categories", "source", "root")

    def __init__(
        self,
        documents: List[Document],
        categories: List[str],
        source: str,
        root: str = "",
    ) -> None:
        self.documents = documents
        self.categories = categories
        self.source = source
        self.root = root

    def __len__(self) -> int:
        """Number of documents.

        Complexity: O(1).
        """
        return len(self.documents)

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return (
            f"NewsgroupsCorpus(docs={len(self.documents)}, "
            f"categories={len(self.categories)}, source={self.source!r})"
        )


def _detect_layout(root: str) -> Optional[Tuple[str, List[str]]]:
    """Find a supported layout under ``root``.

    Args:
        root: Directory that may contain a layout.

    Returns:
        ``(layout_name, split_dirs)`` or ``None`` when nothing matches.

    Complexity:
        Time O(c) for c candidate entries; space O(1).
    """
    if not os.path.isdir(root):
        return None
    entries = sorted(os.listdir(root))
    bydate = [name for name in entries if name.startswith("20news-bydate-")]
    if bydate:
        return "20news-bydate", [os.path.join(root, name) for name in bydate]
    if "20news-18828" in entries:
        return "20news-18828", [os.path.join(root, "20news-18828")]
    # Allow pointing straight at a category-parent directory.
    for name in entries:
        if os.path.isdir(os.path.join(root, name)) and _looks_like_category(
            os.path.join(root, name)
        ):
            return "20news-18828", [root]
    return None


def _looks_like_category(path: str) -> bool:
    """Heuristic: a category directory holds only numbered message files.

    Args:
        path: Directory to inspect.

    Returns:
        ``True`` when the first few entries are digit-named files.

    Complexity:
        Time O(1) (peeks at up to 3 entries).
    """
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    if not entries:
        return False
    checked = 0
    for name in entries:
        if name.isdigit():
            checked += 1
            if checked >= 3:
                return True
    return False


def parse_newsgroups_document(raw: str) -> Tuple[Dict[str, str], str]:
    """Split a raw post into headers and a cleaned body.

    Args:
        raw: File contents.

    Returns:
        ``(headers, body)`` where headers are lower-cased names and the body
        has quoted lines, attribution lines and signatures removed.

    Complexity:
        Time O(len(raw)); space O(len(raw)).
    """
    headers: Dict[str, str] = {}
    cut = raw.find("\n\n")
    if cut < 0:
        header_block, body = raw, ""
    else:
        header_block, body = raw[:cut], raw[cut + 2 :]

    current_key: Optional[str] = None
    for line in header_block.split("\n"):
        if not line.strip():
            continue
        if line[0] in " \t" and current_key is not None:
            headers[current_key] = (headers[current_key] + " " + line.strip()).strip()
            continue
        colon = line.find(":")
        if colon <= 0:
            continue
        key = line[:colon].strip().lower()
        current_key = key if key in _WANTED_HEADERS else None
        if current_key is not None:
            headers[current_key] = line[colon + 1 :].strip()

    return headers, clean_newsgroups_body(body)


def clean_newsgroups_body(body: str) -> str:
    """Drop quoted replies, attributions and signatures from a post body.

    Args:
        body: Raw body text.

    Returns:
        Cleaned body.

    Complexity:
        Time O(len(body)); space O(len(body)).
    """
    kept: List[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if stripped.startswith("|") and len(stripped) > 1:
            continue  # some clients quote with a pipe
        lower = stripped.lower()
        if lower.endswith("writes:") or lower.endswith("wrote:"):
            continue
        if lower.startswith("in article <"):
            continue
        if line.startswith(_SIG_DELIMITER):
            break
        kept.append(line)
    cleaned = "\n".join(kept)
    # Collapse the long runs of blank lines left behind.
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def load_newsgroups(
    root: Optional[str] = None,
    limit: Optional[int] = None,
    per_category: Optional[int] = None,
) -> NewsgroupsCorpus:
    """Load the 20 Newsgroups corpus from disk (or synthesise a fallback).

    Args:
        root: Directory containing the extracted dataset.  Defaults to the
            ``LEXGRAPH_NEWSGROUPS`` environment variable, then
            ``data/raw/20newsgroups`` under the project.
        limit: Keep at most this many documents overall.
        per_category: Keep at most this many documents per category.

    Returns:
        A :class:`NewsgroupsCorpus`.

    Complexity:
        Time O(total bytes read) for a full load; space O(total bytes kept).
        Documents are ordered by ``(category, filename)`` so that a smaller
        ``limit`` yields a strict prefix of a larger one -- benchmarks at 1k,
        2k, 5k, 10k and 18k documents therefore measure nested corpora.
    """
    import data.synthetic as synthetic  # local import keeps loader deps light

    candidates: List[str] = []
    if root:
        candidates.append(root)
    env_root = os.environ.get("LEXGRAPH_NEWSGROUPS")
    if env_root:
        candidates.append(env_root)
    from config import DATA_DIR

    candidates.append(os.path.join(str(DATA_DIR), "20newsgroups"))
    candidates.append(str(DATA_DIR))

    layout: Optional[Tuple[str, List[str]]] = None
    used_root = ""
    for candidate in candidates:
        layout = _detect_layout(candidate)
        if layout is not None:
            used_root = candidate
            break
    if layout is None:
        corpus = synthetic.generate_newsgroups_corpus(seed=7, per_category=per_category or 40)
        return NewsgroupsCorpus(corpus, sorted({d.label for d in corpus}), "synthetic", "")

    layout_name, split_dirs = layout
    documents: List[Document] = []
    categories: List[str] = []
    seen_categories: Dict[str, bool] = {}
    next_id = 0

    for split_dir in split_dirs:
        split_name = os.path.basename(split_dir)
        category_names = sorted(
            name for name in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, name))
        )
        for category in category_names:
            if category not in seen_categories:
                seen_categories[category] = True
                categories.append(category)
            category_path = os.path.join(split_dir, category)
            filenames = sorted(os.listdir(category_path), key=_numeric_sort_key)
            taken = 0
            for filename in filenames:
                if per_category is not None and taken >= per_category:
                    break
                if limit is not None and len(documents) >= limit:
                    break
                path = os.path.join(category_path, filename)
                try:
                    with open(path, "r", encoding="latin-1", errors="replace") as handle:
                        raw = handle.read(MAX_DOCUMENT_CHARS * 3)
                except OSError:
                    continue
                headers, body = parse_newsgroups_document(raw)
                if len(body) < 20:
                    continue
                body = body[:MAX_DOCUMENT_CHARS]
                subject = headers.get("subject", "")
                documents.append(
                    Document(
                        doc_id=next_id,
                        external_id=f"{split_name}/{category}/{filename}",
                        text=body,
                        title=subject or f"{category}/{filename}",
                        label=category,
                        source="newsgroups",
                        snippet=make_snippet(body),
                    )
                )
                next_id += 1
                taken += 1
            if limit is not None and len(documents) >= limit:
                break
        if limit is not None and len(documents) >= limit:
            break

    categories.sort()
    return NewsgroupsCorpus(documents, categories, layout_name, used_root)


def _numeric_sort_key(name: str) -> Tuple[int, str]:
    """Sort message filenames numerically when they are all digits.

    Args:
        name: File name.

    Returns:
        Key tuple; digit-named files sort before other names.

    Complexity:
        Time O(len(name)); space O(1).
    """
    if name.isdigit():
        return (0, f"{int(name):012d}")
    return (1, name)


def split_train_test(
    corpus: NewsgroupsCorpus,
) -> Tuple[List[Document], List[Document]]:
    """Split an already-loaded corpus into train/test by the original split tag.

    Args:
        corpus: A corpus loaded from a bydate layout.

    Returns:
        ``(train, test)``; everything lands in ``train`` for a single-split
        or synthetic corpus.

    Complexity:
        Time O(n); space O(n).
    """
    train: List[Document] = []
    test: List[Document] = []
    for document in corpus.documents:
        if document.external_id.startswith("20news-bydate-test"):
            test.append(document)
        else:
            train.append(document)
    if not test:
        return train, []
    return train, test


__all__ = [
    "NewsgroupsCorpus",
    "load_newsgroups",
    "parse_newsgroups_document",
    "clean_newsgroups_body",
    "split_train_test",
]
