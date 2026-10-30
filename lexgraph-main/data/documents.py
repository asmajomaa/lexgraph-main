"""Shared document model used by every loader and by the whole engine.

Keeping the model in one place means the index, the agent, the web API and the
benchmarks all speak the same vocabulary: a :class:`Document` is a text plus
the handful of fields retrieval and presentation need.
"""

from __future__ import annotations

from typing import List, Optional


class Document:
    """One indexable unit.

    Attributes:
        doc_id: Stable integer identifier used everywhere in the index.
        external_id: Human-readable id (``"newsgroups/comp.graphics/38271"``,
            ``"06_1"`` for a legal case, ``"syn-0007"`` for synthetic data).
        text: The body that was tokenised for the index.
        title: Subject line or case name, shown in results.
        label: Ground-truth class -- newsgroup name or case-year bucket.
        source: ``"newsgroups"``, ``"legal"`` or ``"synthetic"``.
        snippet: A short excerpt for the results pane.

    Space: O(len(text) + len(title) + len(snippet)).
    """

    __slots__ = ("doc_id", "external_id", "text", "title", "label", "source", "snippet")

    def __init__(
        self,
        doc_id: int,
        external_id: str,
        text: str,
        title: str = "",
        label: str = "",
        source: str = "",
        snippet: str = "",
    ) -> None:
        self.doc_id = doc_id
        self.external_id = external_id
        self.text = text
        self.title = title
        self.label = label
        self.source = source
        self.snippet = snippet

    def to_dict(self) -> dict:
        """Plain-dict projection for JSON APIs and the cache.

        Returns:
            Field mapping.

        Complexity:
            Time O(len(text)); space O(len(text)).
        """
        return {
            "doc_id": self.doc_id,
            "external_id": self.external_id,
            "text": self.text,
            "title": self.title,
            "label": self.label,
            "source": self.source,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Document":
        """Rebuild from :meth:`to_dict` output.

        Args:
            payload: Field mapping.

        Returns:
            A :class:`Document`.

        Complexity:
            Time O(len(text)); space O(len(text)).
        """
        return cls(
            doc_id=int(payload["doc_id"]),
            external_id=str(payload["external_id"]),
            text=str(payload["text"]),
            title=str(payload.get("title", "")),
            label=str(payload.get("label", "")),
            source=str(payload.get("source", "")),
            snippet=str(payload.get("snippet", "")),
        )

    def __repr__(self) -> str:
        """Compact debug representation.

        Complexity: O(1).
        """
        return f"Document({self.external_id!r}, label={self.label!r}, chars={len(self.text)})"


def make_snippet(text: str, limit: int = 220) -> str:
    """Collapse whitespace and cut ``text`` to roughly ``limit`` characters.

    Args:
        text: Raw text.
        limit: Maximum snippet length.

    Returns:
        A single-line excerpt, ellipsised when truncated.

    Complexity:
        Time O(min(len(text), limit * len(text) / body)) bounded by the input
        scan; implemented with a small hand-rolled loop over characters to
        avoid regex overhead on every hit.
    """
    if limit <= 0:
        return ""
    parts: List[str] = []
    length = 0
    previous_space = True
    for char in text:
        if char in " \t\r\n":
            if previous_space:
                continue
            parts.append(" ")
            length += 1
            previous_space = True
        else:
            parts.append(char)
            length += 1
            previous_space = False
        if length >= limit:
            break
    snippet = "".join(parts).strip()
    if len(snippet) >= limit and len(text) > limit:
        snippet += "..."
    return snippet


def first_nonempty(*values: Optional[str]) -> str:
    """Return the first non-empty string among ``values``.

    Args:
        *values: Candidate strings.

    Returns:
        The first truthy value after stripping, or ``""``.

    Complexity:
        Time O(len(value)) over the first non-empty candidate; space O(1).
    """
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


__all__ = ["Document", "make_snippet", "first_nonempty"]
