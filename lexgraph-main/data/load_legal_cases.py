"""Legal Case Reports loader (Galgani, 2010 -- UCI "Legal Case Reports").

The published archive ships three parallel XML views per case::

    fulltext/<id>.xml          <name>, <AustLII>, <catchphrases>, <sentences>
    citations_class/<id>.xml   <citations><citation> -> <class>, <tocase>, <AustLII>, <text>
    citations_summ/<id>.xml    <citphrases><citphrase type=cited|citing from="[2005] FCA 1587">

Two format quirks drive the implementation:

1. The files are **not well-formed XML**.  Attributes are written without a
   name and without quotes: ``<catchphrase "id=c0">`` and
   ``<citphrase id="cp0.0" type=cited from="[2005] FCA 1587">``.  A conforming
   parser (``xml.etree``) raises on both, so this module parses with anchored
   regular expressions instead.  That is a deliberate, documented trade-off.

2. Cross-document links are *strings*, not ids.  ``<AustLII>`` URLs contain a
   stray double slash on the cited side, and most cited cases are not in the
   local 3890-case slice at all.  Edges are therefore resolved by the neutral
   citation key ``"[2006] FCA 1"`` extracted from ``<name>`` / ``<tocase>``,
   which matches far more often than URL string equality (1707 vs 1107 of
   25 256 citation records).

The loader emits:

* one :class:`Document` per case, text = catchphrases repeated
  ``CATCHPHRASE_BOOST`` times followed by the case sentences;
* one :class:`LegalCorpus` carrying case names, citation keys, aliases and the
  resolved ``(citing, cited, class)`` edge list.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from data.documents import Document, make_snippet

#: Catchphrase text is repeated this many times in the indexed body so that the
#: author-supplied summary dominates over the long tail of judgment prose.
CATCHPHRASE_BOOST: int = 3

#: Matches the neutral citation inside a case name: ``[2006] FCA 1``.
_CITATION_KEY_RE = re.compile(r"\[\s*(\d{4})\s*\]\s*([A-Za-z][A-Za-z0-9]*)\s*(\d+)")

_NAME_RE = re.compile(r"<name>(.*?)</name>", re.S)
_AUSTLII_RE = re.compile(r"<AustLII>(.*?)</AustLII>", re.S)
# The opening tags come in two malformed flavours in the corpus:
#   <catchphrase "id=c0">      (attribute name and quotes lost)
#   <sentence id="s0">         (well-formed attribute)
# so the optional attribute run must accept a bare quoted token as well as
# name="value" pairs.  ``[^>]*`` would look right but breaks on the bare-quote
# form because the tag body itself contains no ``>`` yet the lazy open/close
# pair still has to start at the right place; anchoring on the tag name is what
# makes both forms work.
_CATCHPHRASE_RE = re.compile(r"<catchphrase\b[^>]*>(.*?)</catchphrase>", re.S)
_SENTENCE_RE = re.compile(r"<sentence\b[^>]*>(.*?)</sentence>", re.S)
_CITATION_BLOCK_RE = re.compile(r"<citation\b[^>]*>(.*?)</citation>", re.S)
_CLASS_RE = re.compile(r"<class>(.*?)</class>", re.S)
_TOCASE_RE = re.compile(r"<tocase>(.*?)</tocase>", re.S)
_CITPHRASE_RE = re.compile(
    r"<citphrase\b[^>]*?type\s*=\s*(\w+)[^>]*?from\s*=\s*\"([^\"]*)\"",
    re.S,
)
_WHITESPACE_RE = re.compile(r"\s+")

#: Citation classes seen in the corpus, in the order the report lists them.
KNOWN_CLASSES: Tuple[str, ...] = (
    "cited",
    "referred to",
    "applied",
    "followed",
    "considered",
    "discussed",
    "distinguished",
    "quoted",
    "affirmed",
    "related",
    "approved",
    "notfollowed",
    "reversed",
    "explained",
    "questioned",
    "ref to",
    "disapproved",
    "noted",
    "compared",
    "referred",
    "doubted",
)


def citation_key(name: str) -> Optional[str]:
    """Extract the neutral citation key from a case name.

    Args:
        name: Case name such as
            ``"Sharman Networks Ltd v Universal Music ... [2006] FCA 1 (5 Jan)"``.

    Returns:
        ``"FCA2006/1"``, or ``None`` when no bracketed citation is present.

    Complexity:
        Time O(len(name)); space O(1).
    """
    match = _CITATION_KEY_RE.search(name or "")
    if not match:
        return None
    year, court, number = match.group(1), match.group(2), match.group(3)
    return f"{court.upper()}{year}/{int(number)}"


def normalise_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and strip.

    Args:
        text: Raw text.

    Returns:
        Cleaned text.

    Complexity:
        Time O(len(text)); space O(len(text)).
    """
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def unescape_entities(text: str) -> str:
    """Decode the five XML entities that appear in the corpus.

    The source files are regex-parsed rather than fed to an XML parser, so
    ``&amp;`` and friends reach us undecoded.  ``html.unescape`` would also
    decode numeric entities, but it is imported here as a single call and only
    the five named entities the data actually contains are handled, which keeps
    the transformation explicit and auditable.

    Args:
        text: Text possibly containing entities.

    Returns:
        Text with ``&amp; &lt; &gt; &quot; &apos;`` decoded.

    Complexity:
        Time O(len(text) * 5) worst case; space O(len(text)).
    """
    if "&" not in text:
        return text
    for entity, replacement in (
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&apos;", "'"),
        ("&amp;", "&"),
    ):
        if entity in text:
            text = text.replace(entity, replacement)
    return text


def strip_markup(text: str) -> str:
    """Remove leftover ``<sentence ...>``/``<catchphrase ...>`` tags from a field.

    Args:
        text: Text that may still contain the source tags.

    Returns:
        Tag-free text.

    Complexity:
        Time O(len(text)); space O(len(text)).
    """
    if "<" not in text:
        return text
    out: list[str] = []
    inside = False
    for char in text:
        if char == "<":
            inside = True
        elif char == ">" and inside:
            inside = False
        elif not inside:
            out.append(char)
    return "".join(out)


def _read(path: str) -> str:
    """Read a legal XML file with a permissive encoding.

    Args:
        path: File path.

    Returns:
        File contents (empty string on failure).

    Complexity:
        Time O(len(file)); space O(len(file)).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def parse_fulltext(raw: str) -> Tuple[str, str, List[str], str]:
    """Parse a ``fulltext/*.xml`` document.

    Args:
        raw: File contents.

    Returns:
        ``(name, austlii_url, catchphrases, body_text)``.

    Complexity:
        Time O(len(raw)); space O(len(raw)).
    """
    name_match = _NAME_RE.search(raw)
    name = unescape_entities(normalise_whitespace(name_match.group(1))) if name_match else ""
    url_match = _AUSTLII_RE.search(raw)
    url = normalise_whitespace(url_match.group(1)) if url_match else ""
    catchphrases = [
        unescape_entities(normalise_whitespace(c)) for c in _CATCHPHRASE_RE.findall(raw)
    ]
    catchphrases = [c for c in catchphrases if c]
    sentences = [
        unescape_entities(normalise_whitespace(s)) for s in _SENTENCE_RE.findall(raw)
    ]
    body = " ".join(s for s in sentences if s)
    return name, url, catchphrases, body


def parse_citation_classes(raw: str) -> List[Tuple[str, Optional[str]]]:
    """Parse ``citations_class/*.xml`` into ``(tocase_name, class)`` records.

    Args:
        raw: File contents.

    Returns:
        One record per ``<citation>`` element.

    Complexity:
        Time O(len(raw)); space O(number of citations).
    """
    records: List[Tuple[str, Optional[str]]] = []
    for block in _CITATION_BLOCK_RE.findall(raw):
        class_match = _CLASS_RE.search(block)
        tocase_match = _TOCASE_RE.search(block)
        tocase = normalise_whitespace(tocase_match.group(1)) if tocase_match else ""
        cited_class = (
            normalise_whitespace(class_match.group(1)).lower() if class_match else None
        )
        if tocase:
            records.append((tocase, cited_class))
    return records


def parse_citing_references(raw: str) -> List[str]:
    """Parse ``citations_summ/*.xml`` for cases that cite this one.

    Args:
        raw: File contents.

    Returns:
        Citation keys (or raw names) of *citing* cases.

    Complexity:
        Time O(len(raw)); space O(number of citing records).
    """
    references: List[str] = []
    for ctype, origin in _CITPHRASE_RE.findall(raw):
        if ctype.strip().lower() == "citing":
            references.append(normalise_whitespace(origin))
    return references


class LegalCorpus:
    """Legal cases plus the resolved citation graph edges.

    Attributes:
        documents: Indexable case documents.
        edges: ``(citing_id, cited_id, class)`` triples with integer doc ids.
        name_by_id: ``doc_id -> case name``.
        key_by_id: ``doc_id -> neutral citation key``.
        aliases: ``alias -> doc_id`` for query-time case resolution.
        source: ``"legal-case-reports"`` or ``"synthetic"``.
        class_counts: ``class -> edge count`` before id resolution.
        citation_records: Total ``<citation>`` records read.
        unresolved: Records whose cited case is outside the local slice.
    """

    __slots__ = (
        "documents",
        "edges",
        "name_by_id",
        "key_by_id",
        "aliases",
        "source",
        "class_counts",
        "citation_records",
        "unresolved",
        "_alias_keys",
    )

    def __init__(
        self,
        documents: List[Document],
        edges: List[Tuple[int, int, str]],
        name_by_id: Dict[int, str],
        key_by_id: Dict[int, str],
        source: str,
    ) -> None:
        self.documents = documents
        self.edges = edges
        self.name_by_id = name_by_id
        self.key_by_id = key_by_id
        self.source = source
        self.class_counts: Dict[str, int] = {}
        self.citation_records = 0
        self.unresolved = 0
        self.aliases: Dict[str, int] = {}
        self._alias_keys: List[Tuple[str, int]] = []

    def build_aliases(self) -> None:
        """Register every query-friendly alias for a case.

        Aliases added, in priority order:
        ``<filename id>``, the normalised case name, the neutral citation key
        (``"fca2006/1"``), and the numeric file suffix (``"1"``).

        Complexity:
            Time O(n); space O(n) alias entries.
        """
        aliases: Dict[str, int] = {}
        ordered: List[Tuple[str, int]] = []
        for document in self.documents:
            candidates = [document.external_id.lower()]
            name = self.name_by_id.get(document.doc_id, "")
            if name:
                candidates.append(normalise_whitespace(name).lower())
            key = self.key_by_id.get(document.doc_id, "")
            if key:
                candidates.append(key.lower())
            for alias in candidates:
                if alias and alias not in aliases:
                    aliases[alias] = document.doc_id
                    ordered.append((alias, document.doc_id))
        self.aliases = aliases
        self._alias_keys = ordered

    def resolve_case(self, needle: str) -> Optional[int]:
        """Best-effort resolution of a user-supplied case reference.

        Tries, in order: exact alias (lower-cased), neutral citation key
        extracted from the needle, then a unique substring match on case names.

        Args:
            needle: Text the user typed, e.g. ``"06_1"``,
                ``"[2006] FCA 1"``, ``"Sharman Networks"`` or ``"FCA 1"``.

        Returns:
            The matching ``doc_id`` or ``None``.

        Complexity:
            Time O(1) for exact hits; O(n) worst case for the substring scan;
            space O(1) beyond the scan.
        """
        text = normalise_whitespace(needle).lower()
        if not text:
            return None
        direct = self.aliases.get(text)
        if direct is not None:
            return direct
        key = citation_key(needle)
        if key is not None and key.lower() in self.aliases:
            return self.aliases[key.lower()]
        # Bare numeric ids like "06_1" survive the exact alias; also try the
        # bracketed form the user might type without the year.
        stripped = text.strip("[]() .,;:'\"")
        direct = self.aliases.get(stripped)
        if direct is not None:
            return direct
        # Last resort: a distinctive fragment of a case name.  Two guards keep
        # this from seeding on an ordinary English word -- the failure it used
        # to have, where "with" resolved to case 4 and "appeal" to case 627:
        #
        # 1. the fragment must match at a *word boundary*, so "appeal" cannot
        #    match inside "appealed" and "art" cannot match inside "article";
        # 2. it must be unique.  An ambiguous fragment returns None so the
        #    agent falls back to its logged BM25 seed instead of guessing.
        if len(stripped) < 4:
            return None
        pattern = re.compile(r"\b" + re.escape(stripped) + r"\b")
        match: Optional[int] = None
        for alias, doc_id in self._alias_keys:
            if not pattern.search(alias):
                continue
            if match is not None and match != doc_id:
                return None  # ambiguous
            match = doc_id
        return match

    def __len__(self) -> int:
        """Number of cases.

        Complexity: O(1).
        """
        return len(self.documents)

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return (
            f"LegalCorpus(cases={len(self.documents)}, edges={len(self.edges)}, "
            f"source={self.source!r})"
        )


def load_legal_cases(
    root: Optional[str] = None,
    limit: Optional[int] = None,
    allow_synthetic: bool = True,
) -> LegalCorpus:
    """Load the Legal Case Reports corpus from disk (or synthesise a fallback).

    Args:
        root: Directory containing ``fulltext/``, ``citations_class/`` and
            ``citations_summ/``.  Defaults to ``LEXGRAPH_LEGAL`` then
            ``data/raw/legal-case-reports`` then ``data/raw/corpus``.
        limit: Keep at most this many cases (alphabetical by file name).
        allow_synthetic: When the corpus is missing, generate a synthetic one
            instead of raising.

    Returns:
        A :class:`LegalCorpus` with edges resolved to doc ids.

    Raises:
        FileNotFoundError: If no corpus is found and ``allow_synthetic`` is
            false.

    Complexity:
        Time O(total bytes of the three XML views); space O(kept text).
        Parsing is linear in file size; edge resolution is O(records) via an
        alias map built once.
    """
    from config import DATA_DIR

    candidates: List[str] = []
    if root:
        candidates.append(root)
    env_root = os.environ.get("LEXGRAPH_LEGAL")
    if env_root:
        candidates.append(env_root)
    candidates.append(os.path.join(str(DATA_DIR), "legal-case-reports"))
    candidates.append(os.path.join(str(DATA_DIR), "corpus"))

    corpus_root = ""
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "fulltext")):
            corpus_root = candidate
            break

    if not corpus_root:
        if not allow_synthetic:
            raise FileNotFoundError("Legal Case Reports corpus not found")
        import data.synthetic as synthetic

        return synthetic.generate_legal_corpus(seed=11)

    fulltext_dir = os.path.join(corpus_root, "fulltext")
    class_dir = os.path.join(corpus_root, "citations_class")
    summ_dir = os.path.join(corpus_root, "citations_summ")

    filenames = sorted(name for name in os.listdir(fulltext_dir) if name.endswith(".xml"))
    if limit is not None:
        filenames = filenames[:limit]

    documents: List[Document] = []
    name_by_id: Dict[int, str] = {}
    key_by_id: Dict[int, str] = {}
    id_by_key: Dict[str, int] = {}
    id_by_name: Dict[str, int] = {}

    for doc_id, filename in enumerate(filenames):
        case_id = filename[: -len(".xml")]
        raw = _read(os.path.join(fulltext_dir, filename))
        name, _url, catchphrases, body = parse_fulltext(raw)
        title = name or case_id
        boosted = " ".join(catchphrases * CATCHPHRASE_BOOST)
        text = (boosted + " " + body).strip() if boosted else body
        label = citation_key(title) or case_id
        documents.append(
            Document(
                doc_id=doc_id,
                external_id=case_id,
                text=text,
                title=title,
                label=label.split("/")[0][:4] if label else "",
                source="legal",
                snippet=make_snippet(body or boosted),
            )
        )
        name_by_id[doc_id] = title
        key = citation_key(title)
        if key:
            key_by_id[doc_id] = key
            id_by_key.setdefault(key, doc_id)
        id_by_name.setdefault(normalise_whitespace(title).lower(), doc_id)

    corpus = LegalCorpus(documents, [], name_by_id, key_by_id, "legal-case-reports")
    corpus.build_aliases()

    edge_set: Dict[str, Tuple[int, int, str]] = {}
    class_counts: Dict[str, int] = {}
    unresolved = 0
    records = 0

    if os.path.isdir(class_dir):
        for filename in filenames:
            case_id = filename[: -len(".xml")]
            source_id = corpus.aliases.get(case_id.lower())
            if source_id is None:
                continue
            raw = _read(os.path.join(class_dir, filename))
            for tocase, cited_class in parse_citation_classes(raw):
                records += 1
                label = cited_class or "cited"
                class_counts[label] = class_counts.get(label, 0) + 1
                key = citation_key(tocase)
                target_id = id_by_key.get(key) if key else None
                if target_id is None:
                    target_id = id_by_name.get(normalise_whitespace(tocase).lower())
                if target_id is None or target_id == source_id:
                    unresolved += 1
                    continue
                unique = f"{source_id}\x00{target_id}"
                if unique not in edge_set:
                    edge_set[unique] = (source_id, target_id, label)

    if os.path.isdir(summ_dir):
        for filename in filenames:
            case_id = filename[: -len(".xml")]
            target_id = corpus.aliases.get(case_id.lower())
            if target_id is None:
                continue
            raw = _read(os.path.join(summ_dir, filename))
            for origin in parse_citing_references(raw):
                key = citation_key(origin)
                source_id = id_by_key.get(key) if key else None
                if source_id is None:
                    source_id = id_by_name.get(normalise_whitespace(origin).lower())
                if source_id is None or source_id == target_id:
                    continue
                unique = f"{source_id}\x00{target_id}"
                if unique not in edge_set:
                    # ``citations_summ`` carries no treatment class; the neutral
                    # default is the generic "cited" relation.
                    edge_set[unique] = (source_id, target_id, "cited")

    corpus.edges = list(edge_set.values())
    corpus.class_counts = class_counts
    corpus.citation_records = records
    corpus.unresolved = unresolved
    return corpus


__all__ = [
    "LegalCorpus",
    "load_legal_cases",
    "citation_key",
    "parse_fulltext",
    "parse_citation_classes",
    "parse_citing_references",
    "normalise_whitespace",
    "CATCHPHRASE_BOOST",
    "KNOWN_CLASSES",
]
