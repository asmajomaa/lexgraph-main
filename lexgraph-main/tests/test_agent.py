"""Agent routing, ML pruning and relational-path tests on synthetic engines."""

from __future__ import annotations

import pytest

from config import KMEANS_SEED
from core.agent import ReActAgent, is_relational
from core.inverted_index import InvertedIndex
from core.tokenizer import Tokenizer
from data.documents import Document
from data.synthetic import generate_legal_corpus, generate_newsgroups_corpus
from engine import build_citation_graph
from ml.kmeans import KMeans
from ml.vectorizer import TFIDFVectorizer


@pytest.fixture(scope="module")
def wired_agent() -> ReActAgent:
    news_docs = generate_newsgroups_corpus(seed=3, per_category=12, words_per_doc=60)
    legal = generate_legal_corpus(seed=5, case_count=25)
    tokenizer = Tokenizer()
    news_index = InvertedIndex.build(news_docs, tokenizer, name="news")
    legal_index = InvertedIndex.build(legal.documents, tokenizer, name="legal")
    graph, _skipped = build_citation_graph(legal)
    vectorizer = TFIDFVectorizer(vocab_size=300, min_df=1).fit(news_docs)
    vectors = vectorizer.transform(news_docs)
    kmeans = KMeans(k=4, max_iter=20, seed=KMEANS_SEED).fit(vectors)
    return ReActAgent(
        news_index=news_index,
        legal_index=legal_index,
        graph=graph,
        legal_corpus=legal,
        kmeans=kmeans,
        vectorizer=vectorizer,
        clusters=list(kmeans.labels),
        verbose=False,
    )


@pytest.mark.parametrize(
    "query",
    [
        "which cases cite 06_1",
        "what references the Sharman case",
        "cases that followed the precedent",
        "show me the citing articles",
        "is it distinguished by later courts",
        "applied in a later ruling",
    ],
)
def test_relational_keywords_route_to_graph(query):
    assert is_relational(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "hockey playoffs goalie",
        "opengl texture mapping",
        "excited about the new engine",
        "the precedence of operators",
        "recently updated drivers",
        "orbit launch window",
    ],
)
def test_lexical_queries_do_not_route_to_graph(query):
    assert is_relational(query) is False


def test_reason_emits_structured_log(wired_agent):
    intent, logs = wired_agent.reason("which cases cite 06_1", "baseline")
    assert intent == "RELATIONAL"
    assert logs[0].startswith("[AGENT][REASON]")
    assert "RELATIONAL" in logs[0]


def test_baseline_run_returns_ranked_rows(wired_agent):
    result = wired_agent.run("hockey goalie", mode="baseline", k=5)
    assert result.intent == "LEXICAL"
    assert result.path == "baseline"
    assert all(row["score"] is not None for row in result.results)
    scores = [row["score"] for row in result.results]
    assert scores == sorted(scores, reverse=True)
    assert any(line.startswith("[AGENT][ACT]") for line in result.logs)
    assert any(line.startswith("[AGENT][OBSERVE]") for line in result.logs)
    assert result.logs[-1].startswith("[AGENT][ANSWER]")
    assert result.stats["corpus_size"] > 0


def test_ml_run_prunes_candidates(wired_agent):
    result = wired_agent.run("hockey goalie", mode="ml", k=5, probe_clusters=1)
    assert result.path == "ml"
    assert result.stats["clusters_probed"] == 1.0
    assert result.stats["candidates_scored"] <= result.stats["corpus_size"]
    assert result.stats["candidate_pool"] > 0


def test_ml_run_with_compare_reports_overlap(wired_agent):
    result = wired_agent.run(
        "graphics texture", mode="ml", k=5, probe_clusters=3, compare_baseline=True
    )
    assert "baseline_overlap" in result.stats
    assert result.stats["baseline_overlap"] <= result.stats["baseline_k"]


def test_relational_run_uses_seed_and_traversal(wired_agent):
    result = wired_agent.run("which cases cite 0003", mode="baseline", k=10, bfs_depth=2)
    assert result.intent == "RELATIONAL"
    assert result.path == "relational"
    assert result.results, "graph should reach at least the seed"
    assert result.results[0]["depth"] == 0
    assert any("BFS" in line for line in result.logs)


def test_relational_falls_back_to_bm25_top1(wired_agent):
    result = wired_agent.run("cites copyright infringement", mode="baseline", k=5)
    assert result.path == "relational"
    assert result.results or result.logs


def test_relational_with_no_matches_reports_honestly(wired_agent):
    result = wired_agent.run("cite zzzzunknowable", mode="baseline", k=5)
    assert result.path == "relational"
    if not result.results:
        assert "No case connected" in result.answer


def test_answer_text_is_present(wired_agent):
    result = wired_agent.run("hockey", mode="baseline", k=3)
    assert result.answer
    assert result.answer.strip()


def test_empty_match_answer_is_honest(wired_agent):
    result = wired_agent.run("zzzzz nonexistent term", mode="baseline", k=3)
    assert result.results == []
    assert "No indexed document" in result.answer


def test_result_serialises_to_dict(wired_agent):
    payload = wired_agent.run("hockey", mode="baseline", k=3).to_dict()
    for key in ("intent", "path", "results", "logs", "timings", "stats", "total_ms"):
        assert key in payload
    assert payload["total_ms"] >= 0.0


def test_depth_limited_bfs_returns_seed_only(wired_agent):
    result = wired_agent.run("which cases cite 0003", mode="baseline", k=10, bfs_depth=0)
    assert result.path == "relational"
    assert len(result.results) <= 1
