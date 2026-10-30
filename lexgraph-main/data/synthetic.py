"""Synthetic fallback corpora and graph fixtures.

The real datasets are present for this build, but a project that *only* runs
when a 100 MB download succeeds is fragile.  These generators produce corpora
that satisfy the same contracts as the real loaders (labels, case ids, citation
keys, typed edges) so that tests, demos and the report's structural checks run
anywhere.

Everything here is seeded, so regenerating produces byte-identical corpora.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from data.documents import Document, make_snippet

#: Newsgroup-style categories with distinctive vocabularies.  Each tuple is
#: ``(category, topic words)``; documents are built by sampling topic words.
_SYNTHETIC_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("comp.graphics", ("image", "pixel", "render", "texture", "opengl", "raster", "shading")),
    ("sci.space", ("orbit", "rocket", "nasa", "satellite", "lunar", "launch", "astronaut")),
    ("rec.sport.hockey", ("hockey", "goalie", "puck", "playoff", "skate", "period", "team")),
    ("talk.politics.misc", ("policy", "senate", "election", "vote", "party", "reform", "budget")),
    ("comp.os.linux", ("kernel", "linux", "driver", "module", "compile", "patch", "boot")),
    ("sci.med", ("patient", "clinical", "dose", "symptom", "therapy", "study", "health")),
    ("rec.autos", ("engine", "transmission", "brake", "wheel", "dealer", "mileage", "repair")),
    ("soc.culture.misc", ("culture", "festival", "tradition", "language", "community", "story")),
)

#: Filler words mixed into synthetic newsgroup posts.
_FILLER: Tuple[str, ...] = (
    "the", "and", "with", "that", "this", "from", "have", "about", "would",
    "there", "which", "their", "been", "other", "into", "more", "some", "when",
)


def generate_newsgroups_corpus(
    seed: int = 7,
    per_category: int = 40,
    words_per_doc: int = 120,
) -> List[Document]:
    """Generate a small 20-Newsgroups-shaped corpus.

    Args:
        seed: RNG seed.
        per_category: Documents per synthetic category.
        words_per_doc: Target body length.

    Returns:
        Documents with ``source="synthetic"`` and the category as label.

    Complexity:
        Time O(total words); space O(total words).
    """
    rng = random.Random(seed)
    documents: List[Document] = []
    next_id = 0
    for category, topic_words in _SYNTHETIC_CATEGORIES:
        for index in range(per_category):
            words: List[str] = []
            while len(words) < words_per_doc:
                if rng.random() < 0.45:
                    words.append(rng.choice(topic_words))
                else:
                    words.append(rng.choice(_FILLER))
            body = " ".join(words)
            subject = f"{topic_words[index % len(topic_words)]} question {index}"
            documents.append(
                Document(
                    doc_id=next_id,
                    external_id=f"synthetic/{category}/{index:04d}",
                    text=body,
                    title=subject,
                    label=category,
                    source="synthetic",
                    snippet=make_snippet(body),
                )
            )
            next_id += 1
    return documents


#: The canonical edge classes the synthetic legal corpus draws from.
_SYNTHETIC_CLASSES: Tuple[str, ...] = (
    "cited", "applied", "followed", "distinguished", "considered",
)

#: Legal-flavoured vocabulary for synthetic judgments.
_LEGAL_TOPICS: Tuple[Tuple[str, ...], ...] = (
    ("copyright", "infringement", "authorisation", "licence", "sound recording"),
    ("mareva", "assets", "disclosure", "affidavit", "preservation order"),
    ("negligence", "duty of care", "breach", "causation", "damages"),
    ("contract", "termination", "misrepresentation", "consideration", "clause"),
    ("appeal", "leave", "judgment", "jurisdiction", "grounds"),
)

_LEGAL_FILLER: Tuple[str, ...] = (
    "the", "court", "held", "that", "matter", "proceedings", "applicant",
    "respondent", "order", "submission", "evidence", "section", "federal",
)


def generate_legal_corpus(
    seed: int = 11,
    case_count: int = 60,
    avg_citations: float = 2.5,
) -> "LegalCorpus":
    """Generate a synthetic Legal-Case-Reports-shaped corpus.

    Args:
        seed: RNG seed.
        case_count: Number of cases.
        avg_citations: Mean out-degree.

    Returns:
        A :class:`data.load_legal_cases.LegalCorpus` with ``source="synthetic"``.

    Complexity:
        Time O(case_count * words + edges); space O(same).
    """
    from data.load_legal_cases import LegalCorpus, normalise_whitespace

    rng = random.Random(seed)
    documents: List[Document] = []
    name_by_id: Dict[int, str] = {}
    key_by_id: Dict[str, str] = {}
    aliases: Dict[str, int] = {}

    for index in range(case_count):
        year = 1990 + index % 25
        number = index // 25 + 1
        key = f"FCA{year}/{number}"
        name = f"R v Synthetic {index} [ {year} ] FCA {number}"
        name = normalise_whitespace(name)
        topics = _LEGAL_TOPICS[index % len(_LEGAL_TOPICS)]
        words: List[str] = []
        for _ in range(90):
            words.append(rng.choice(topics) if rng.random() < 0.4 else rng.choice(_LEGAL_FILLER))
        catchphrases = " ".join(rng.choice(topics) for _ in range(6))
        body = " ".join(rng.choice(topics) for _ in range(30)) + " " + " ".join(words)
        text = (catchphrases + " ") * 3 + body
        documents.append(
            Document(
                doc_id=index,
                external_id=f"{index:04d}",
                text=text,
                title=name,
                label=str(year),
                source="legal",
                snippet=make_snippet(body),
            )
        )
        name_by_id[index] = name
        key_by_id[index] = key
        aliases[f"{index:04d}"] = index
        aliases[name.lower()] = index
        aliases[key.lower()] = index

    edges: List[Tuple[int, int, str]] = []
    seen: Dict[str, bool] = {}
    for source in range(case_count):
        out_degree = max(1, int(rng.gauss(avg_citations, 1.0)))
        for _ in range(out_degree):
            target = rng.randrange(case_count)
            if target == source:
                continue
            marker = f"{source}\x00{target}"
            if marker in seen:
                continue
            seen[marker] = True
            edges.append((source, target, rng.choice(_SYNTHETIC_CLASSES)))

    corpus = LegalCorpus(documents, edges, name_by_id, key_by_id, "synthetic")
    corpus.aliases = aliases
    corpus._alias_keys = [(alias, doc_id) for alias, doc_id in aliases.items()]
    corpus.citation_records = len(edges)
    for _source, _target, edge_class in edges:
        corpus.class_counts[edge_class] = corpus.class_counts.get(edge_class, 0) + 1
    return corpus


__all__ = ["generate_newsgroups_corpus", "generate_legal_corpus"]
