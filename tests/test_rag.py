"""Unit tests for the RAG pipeline components — no network calls."""
from __future__ import annotations

import pytest
from research_pipeline.rag.search_tool import SearchTool
from research_pipeline.rag.vector_db import VectorDB
from research_pipeline.rag.conversation import ConversationalRAGAgent
from research_pipeline.rag.ingestion import run_ingestion_pipeline

# ---------------------------------------------------------------------------
# SearchTool
# ---------------------------------------------------------------------------

def test_search_tool_generates_correct_number_of_subqueries():
    tool = SearchTool()
    for n in [3, 5, 7]:
        qs = tool.generate_subqueries("AI in medicine", max_queries=n)
        assert len(qs) == n
    # topic with very few valid terms still returns something
    qs = tool.generate_subqueries("AI", max_queries=5)
    assert len(qs) >= 1


def test_search_tool_subqueries_no_hang_on_sparse_vocabulary():
    """Regression test: topics that yield 0-1 lowercase-matching tokens
    (e.g. mostly capitalized acronyms) must not hang. Previously,
    tokenizing against the *original-case* string (instead of lowercasing
    first) silently dropped capitalized words; with <2 surviving base
    terms, the fill-loop's add_query() calls became permanent no-ops
    while `queries` never reached `max_queries`, causing an infinite loop.
    Guarded here with a hard wall-clock budget so a regression fails fast
    instead of hanging the whole test run."""
    import time
    tool = SearchTool()
    sparse_topics = ["AI in medicine", "AI", "ML", "NASA JPL", "AI ML"]
    for topic in sparse_topics:
        start = time.time()
        qs = tool.generate_subqueries(topic, max_queries=7)
        assert time.time() - start < 2.0, f"generate_subqueries hung on topic={topic!r}"
        assert 1 <= len(qs) <= 7


def test_search_tool_subqueries_reference_topic_terms():
    tool = SearchTool()
    qs = tool.generate_subqueries("machine learning diagnostics", max_queries=4)
    assert all("machine" in q or "learning" in q or "diagnostics" in q for q in qs)


def test_search_tool_collect_documents_respects_max_sources():
    """collect_documents must not return more than max_sources docs.
    Uses a very small max so the mock delay is short (< 2 s total)."""
    tool = SearchTool()
    docs = tool.collect_documents("AI in healthcare", max_sources=2)
    assert len(docs) <= 2
    for doc in docs:
        assert doc.url.startswith("http")
        assert doc.title
        assert doc.text   # full text must be non-empty for chunking


def test_search_tool_uses_realistic_corpus_domains():
    tool = SearchTool(max_days_old=180)
    docs = tool.collect_documents("artificial intelligence", max_sources=3)
    assert len(docs) <= 3
    assert docs
    assert any(domain in docs[0].url for domain in ["techcrunch.com", "sciencedirect.com", "arxiv.org", "nature.com", "reuters.com", "mit.edu"])


def test_search_tool_prefers_live_backend_when_available():
    """When a live search backend is configured and available, its results
    should be used ahead of the local corpus."""
    class StubLiveBackend:
        def is_available(self) -> bool:
            return True

        def search(self, query, max_days_old, top_k=5):
            return [
                {"url": "https://live.example.com/a", "title": "Live result A", "snippet": "Live web content.", "published_days_ago": None},
                {"url": "https://live.example.com/b", "title": "Live result B", "snippet": "More live web content.", "published_days_ago": None},
            ]

    tool = SearchTool(live_backend=StubLiveBackend())
    docs = tool.collect_documents("AI in medicine", max_sources=2)
    assert docs
    assert all(doc.url.startswith("https://live.example.com/") for doc in docs)


def test_search_tool_falls_back_to_corpus_when_live_backend_fails():
    """A live backend that is available but errors on every call must not
    break document collection -- it should fall back to the local corpus."""
    class FailingLiveBackend:
        def is_available(self) -> bool:
            return True

        def search(self, query, max_days_old, top_k=5):
            raise RuntimeError("simulated network failure")

    tool = SearchTool(live_backend=FailingLiveBackend())
    docs = tool.collect_documents("AI in medicine", max_sources=3)
    assert docs
    assert all(not doc.url.startswith("https://live.example.com/") for doc in docs)


def test_search_tool_preserves_archived_url_metadata(monkeypatch):
    tool = SearchTool()

    def fake_search_corpus(query: str, top_k: int):
        return [{
            "url": "https://example.com/missing",
            "title": "Missing article",
            "snippet": "This content is archived locally.",
            "published_days_ago": 10,
            "source_id": "src_missing",
        }]

    monkeypatch.setattr(tool, "_search_corpus", fake_search_corpus)
    monkeypatch.setattr(tool, "_check_and_archive_url", lambda url: (None, "not found", "file:///tmp/archives/src_missing.html"))

    docs = tool.collect_documents("AI in medicine", max_sources=1)
    assert len(docs) == 1
    assert docs[0].url == "https://example.com/missing"
    assert docs[0].url_status is not None
    assert docs[0].url_status["archived_url"] == "file:///tmp/archives/src_missing.html"
    assert docs[0].archived_url == "file:///tmp/archives/src_missing.html"


# ---------------------------------------------------------------------------
# VectorDB
# ---------------------------------------------------------------------------

_DOCS = [
    {"title": "Medical AI", "text": "AI systems assist radiologists by spotting anomalies in medical images.", "url": "https://example.com/med-ai"},
    {"title": "Clinical workflow", "text": "Hospitals use AI to reduce administrative load and improve triage workflows.", "url": "https://example.com/triage"},
]

def test_vector_db_chunks_and_stores_documents():
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    report = db.index_documents(_DOCS, retention_days=30)
    assert report["chunk_count"] > 0
    assert report["new_added"] > 0


def test_vector_db_deduplicates_identical_content():
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    db.index_documents(_DOCS, retention_days=30)
    report2 = db.index_documents(_DOCS, retention_days=30)  # same docs again
    assert report2["duplicate_count"] > 0
    # No new chunks should have been added
    assert report2["new_added"] == 0


def test_vector_db_purge_expired_removes_nothing_for_fresh_docs():
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    db.index_documents(_DOCS, retention_days=30)
    purged = db.purge_expired(retention_days=30)
    assert purged == 0   # just indexed — nothing expired yet


def test_vector_db_batches_embeddings_for_memory_safety(monkeypatch):
    class BatchTrackingModel:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def encode(self, sentences: list[str], convert_to_numpy: bool = True):
            import numpy as np
            self.calls.append(len(sentences))
            return np.ones((len(sentences), 384))

    model = BatchTrackingModel()
    db = VectorDB(chunk_size=10, chunk_overlap=2, max_embedding_batch_size=2)
    db.has_model = True
    db.model = model
    db._chunk_text = lambda text: [text + " 1", text + " 2", text + " 3"]

    db.index_documents([{"title": "A", "text": "one two three four five", "url": "https://example.com"}], retention_days=30)

    assert model.calls == [2, 1]


def test_vector_db_query_returns_ranked_results():
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    db.index_documents(_DOCS, retention_days=30)
    hits = db.query("AI radiology anomaly detection", top_k=2)
    assert len(hits) <= 2
    if len(hits) == 2:
        score1, _ = hits[0]
        score2, _ = hits[1]
        assert score1 >= score2   # descending order


# ---------------------------------------------------------------------------
# ConversationalRAGAgent
# ---------------------------------------------------------------------------

def _populated_db() -> VectorDB:
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    db.index_documents(_DOCS, retention_days=30)
    return db


def test_rag_agent_returns_grounded_answer():
    agent = ConversationalRAGAgent(vector_db=_populated_db())
    result = agent.answer_query(session_id="t1", query="How does AI help in radiology?")
    assert result["grounded"] is True
    assert "Source" in result["response"] or "Based on" in result["response"]
    assert len(result["sources"]) > 0


def test_ingestion_pipeline_reports_partial_when_token_budget_is_hit(tmp_path, monkeypatch):
    from research_pipeline.rag.search_tool import SearchDocument

    class StubSearchTool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def collect_documents(self, topic: str, max_sources: int = 10) -> list[SearchDocument]:
            return [
                SearchDocument(url="https://example.com/one", title="One", text="Alpha beta gamma"),
                SearchDocument(url="https://example.com/two", title="Two", text="Delta epsilon zeta"),
            ]

    class StubVectorDB:
        def __init__(self) -> None:
            self.calls = []

        def index_documents(self, documents: list[dict], retention_days: int = 30) -> dict:
            self.calls.append(documents)
            return {"new_added": len(documents), "duplicate_count": 0, "chunk_count": len(documents), "purged": 0}

    monkeypatch.setattr("research_pipeline.rag.ingestion.SearchTool", StubSearchTool)
    db = StubVectorDB()
    report = run_ingestion_pipeline(topic="test topic", db=db, output_dir=str(tmp_path), token_budget=3)

    assert report["status"] == "PARTIAL"
    assert report["articles_processed"] == 1
    assert report["articles_skipped"] == 0


def test_rag_agent_refuses_ungrounded_query():
    """Empty DB → score below threshold → grounded=False with explicit refusal."""
    agent = ConversationalRAGAgent(vector_db=VectorDB())
    result = agent.answer_query(session_id="t2", query="What is the meaning of life?")
    assert result["grounded"] is False
    assert result["sources"] == []


def test_rag_agent_resolves_follow_up_references(tmp_path):
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    db.index_documents([
        {
            "title": "Radiology AI",
            "text": "AI helps radiologists detect anomalies in medical imaging.",
            "url": "https://example.com/ai-radiology",
            "published_days_ago": 2,
        },
        {
            "title": "Clinical workflow AI",
            "text": "Hospitals use AI triage workflows to reduce administrative burdens.",
            "url": "https://example.com/ai-triage",
            "published_days_ago": 3,
        },
    ], retention_days=30)

    agent = ConversationalRAGAgent(vector_db=db, session_dir=str(tmp_path))

    result1 = agent.answer_query(session_id="followup-test", query="How does AI help in radiology?")
    assert result1["grounded"] is True
    assert len(result1["sources"]) > 0

    result2 = agent.answer_query(session_id="followup-test", query="Tell me more about that")
    assert result2["grounded"] is True
    assert len(result2["sources"]) > 0

    result3 = agent.answer_query(session_id="followup-test", query="What about the second point?")
    assert result3["grounded"] is True
    assert len(result3["sources"]) > 0


def test_rag_agent_does_not_misroute_unrelated_query_with_marker_substring(tmp_path):
    """Regression test: follow-up detection previously used plain substring
    matching (`marker in query.lower()`), so a brand-new, unrelated question
    containing a marker only as part of another word -- e.g. "it" inside
    "capital" -- was wrongly treated as a follow-up. That caused the
    grounding check to run against the *previous* topic's query instead of
    the actual new question, so an out-of-scope question could come back
    grounded=True by riding on unrelated prior context. Word-boundary
    matching must prevent this."""
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    db.index_documents([
        {
            "title": "Radiology AI",
            "text": "AI helps radiologists detect anomalies in medical imaging.",
            "url": "https://example.com/ai-radiology",
            "published_days_ago": 2,
        },
    ], retention_days=30)

    agent = ConversationalRAGAgent(vector_db=db, session_dir=str(tmp_path))

    result1 = agent.answer_query(session_id="marker-substring-test", query="How does AI help in radiology?")
    assert result1["grounded"] is True

    # "capital" contains "it" as a substring but is not a follow-up reference.
    result2 = agent.answer_query(session_id="marker-substring-test", query="What is the capital of France?")
    assert result2["grounded"] is False
    assert result2["sources"] == []


def test_rag_agent_filters_temporal_queries(tmp_path):
    db = VectorDB(chunk_size=10, chunk_overlap=2)
    db.index_documents([
        {
            "title": "Recent AI medical study",
            "text": "New findings in AI-assisted medical imaging were published this week.",
            "url": "https://example.com/recent-ai",
            "published_days_ago": 2,
        },
        {
            "title": "Historical AI medical research",
            "text": "Past AI research from years ago in medical imaging.",
            "url": "https://example.com/old-ai",
            "published_days_ago": 45,
        },
    ], retention_days=30)

    agent = ConversationalRAGAgent(vector_db=db, session_dir=str(tmp_path))
    result = agent.answer_query(session_id="temporal-test", query="What changed in the last week compared to today?")

    assert result["grounded"] is True
    assert all(source["url"] == "https://example.com/recent-ai" for source in result["sources"])
    assert any("Recent AI medical study" in source["title"] for source in result["sources"])


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

def test_reranker_reorders_by_relevance_bm25_fallback():
    """With no cross-encoder weights available, the BM25 fallback should
    still surface the more lexically-relevant chunk first."""
    from research_pipeline.rag.reranker import CrossEncoderReranker
    from research_pipeline.rag.vector_db import Chunk

    reranker = CrossEncoderReranker(model_name="__force_unavailable__")
    reranker.backend = "bm25_fallback"  # force the fallback path deterministically

    c1 = Chunk(id="1", text="radiology anomaly detection using AI in hospitals")
    c2 = Chunk(id="2", text="the weather today is sunny with light wind")
    candidates = [(0.5, c1), (0.5, c2)]  # identical first-pass scores

    reranked = reranker.rerank("AI radiology anomaly detection", candidates, top_k=2)
    assert reranked[0][1].id == "1"  # more relevant chunk ranked first


def test_reranker_falls_back_gracefully_when_model_predict_raises():
    """If the cross-encoder backend errors at prediction time, rerank()
    must still return results via the BM25 fallback rather than raising
    or silently returning an empty list."""
    from research_pipeline.rag.reranker import CrossEncoderReranker
    from research_pipeline.rag.vector_db import Chunk

    class _BrokenModel:
        def predict(self, pairs):
            raise RuntimeError("simulated model failure")

    reranker = CrossEncoderReranker()
    reranker.backend = "cross_encoder"
    reranker.model = _BrokenModel()

    c1 = Chunk(id="1", text="AI radiology detection")
    reranked = reranker.rerank("AI radiology", [(0.5, c1)], top_k=1)
    assert len(reranked) == 1
    assert reranked[0][1].id == "1"


def test_rag_agent_session_persistence(tmp_path):
    """Session history is saved and reloaded correctly."""
    db    = _populated_db()
    agent = ConversationalRAGAgent(vector_db=db, session_dir=str(tmp_path))
    agent.answer_query(session_id="sess-abc", query="What is AI triage?")
    # Re-instantiate to force reload from disk
    agent2 = ConversationalRAGAgent(vector_db=db, session_dir=str(tmp_path))
    history = agent2._load_history("sess-abc")
    assert len(history) >= 2   # user + assistant turn


def test_ingestion_pipeline_forwards_retention_days():
    """Regression test: `run_ingestion_pipeline`'s `retention_days` argument
    must actually reach `VectorDB.index_documents` — it was previously
    accepted but silently discarded in favor of a hardcoded 30, so the
    `--retention-days` CLI flag had no effect at all."""
    db = VectorDB()
    run_ingestion_pipeline(topic="AI in healthcare", db=db, max_sources=3,
                            retention_days=9999)
    assert db.chunks, "expected at least one chunk to be indexed"
    assert all(c.metadata.get("retention_days") == 9999 for c in db.chunks)
