"""Tests for the real-LLM integration (research_pipeline.llm_client and its
wiring into ConversationalRAGAgent).

No real network/API calls are made in these tests -- LLMClient's
availability check and the agent's fallback-on-failure behavior are tested
with a fake client, exactly the contract every call site relies on.

sentence_transformers is mocked the same way test_rag.py does it: without
this, VectorDB falls back to its offline lexical scorer, which does exact
token matching and would score "radiology" (query) vs "radiologists"
(chunk) as zero overlap -- a false negative that has nothing to do with
what this file is actually testing (the LLM wiring). Mocking keeps these
tests deterministic and independent of whether the sandbox running them
has internet access to huggingface.co.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

if not isinstance(getattr(sys.modules.get("sentence_transformers"), "SentenceTransformer", None), type) \
        or getattr(sys.modules.get("sentence_transformers"), "SentenceTransformer", None).__name__ != "MockSentenceTransformer":
    class MockSentenceTransformer:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name
        def encode(self, sentences: list[str], convert_to_numpy: bool = True):
            import numpy as np
            return np.ones((len(sentences), 384))
    sys.modules["sentence_transformers"] = MagicMock()
    import sentence_transformers
    sentence_transformers.SentenceTransformer = MockSentenceTransformer

from research_pipeline.llm_client import LLMClient


def _populated_db():
    """Builds a VectorDB through the real index_documents() path so chunks
    get real embeddings computed -- directly assigning db.chunks (as an
    earlier version of these tests did) skips that step, leaving
    chunk.embedding=None and making every query() return zero results
    regardless of relevance. This mirrors the _populated_db() helper in
    test_rag.py."""
    from research_pipeline.rag.vector_db import VectorDB
    db = VectorDB()
    db.index_documents([
        {"title": "Radiology AI", "text": "AI assists radiologists in anomaly detection.",
         "url": "https://example.com/a"},
    ], retention_days=30)
    return db


def test_llm_client_unavailable_without_api_key():
    client = LLMClient(api_key=None)
    assert client.is_available() is False


def test_llm_client_generate_raises_clear_error_when_unavailable():
    import pytest
    client = LLMClient(api_key=None)
    with pytest.raises(RuntimeError, match="not available"):
        client.generate(system="x", user="y")


def test_llm_client_uses_groq_when_groq_key_present(monkeypatch):
    import research_pipeline.llm_client as llm_client_module

    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    fake_client = object()
    monkeypatch.setattr(llm_client_module, "_build_groq_client", lambda api_key: fake_client)

    client = llm_client_module.LLMClient(api_key=None)

    assert client.is_available() is True
    assert client._client is fake_client


class _FakeLLM:
    """Stand-in for LLMClient in tests -- avoids any real API call."""
    def __init__(self, response_text: str = "The sources confirm this. [1]", should_fail: bool = False):
        self.response_text = response_text
        self.should_fail = should_fail
        self.last_call = None

    def is_available(self) -> bool:
        return True

    def generate(self, system: str, user: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
        self.last_call = {"system": system, "user": user}
        if self.should_fail:
            raise RuntimeError("simulated API failure")
        return self.response_text


def test_rag_agent_uses_real_llm_when_available():
    """When an available LLM client is injected, its output (not the
    template) should be returned as the response."""
    from research_pipeline.rag.conversation import ConversationalRAGAgent

    db = _populated_db()

    fake_llm = _FakeLLM(response_text="AI helps radiologists spot anomalies faster. [1]")
    agent = ConversationalRAGAgent(vector_db=db, session_dir="/tmp/test_llm_sessions", llm=fake_llm)
    result = agent.answer_query(session_id="llm-test-1", query="How does AI help in radiology?")

    assert result["grounded"] is True
    assert result["response"] == "AI helps radiologists spot anomalies faster. [1]"
    assert fake_llm.last_call is not None
    assert "Question: How does AI help in radiology?" in fake_llm.last_call["user"]


def test_rag_agent_falls_back_to_template_when_llm_call_fails():
    """If the real LLM call raises (network error, rate limit, etc.), the
    chat must still answer via the template path rather than crashing."""
    from research_pipeline.rag.conversation import ConversationalRAGAgent

    db = _populated_db()

    failing_llm = _FakeLLM(should_fail=True)
    agent = ConversationalRAGAgent(vector_db=db, session_dir="/tmp/test_llm_sessions", llm=failing_llm)
    result = agent.answer_query(session_id="llm-test-2", query="How does AI help in radiology?")

    assert result["grounded"] is True
    assert "Based on the verified database" in result["response"]  # template fallback signature


def test_rag_agent_uses_template_when_llm_unavailable():
    """Default behavior (no ANTHROPIC_API_KEY) is unchanged from before this
    feature was added: template-based generation."""
    from research_pipeline.rag.conversation import ConversationalRAGAgent

    db = _populated_db()

    agent = ConversationalRAGAgent(vector_db=db, session_dir="/tmp/test_llm_sessions", llm=LLMClient(api_key=None))
    result = agent.answer_query(session_id="llm-test-3", query="How does AI help in radiology?")

    assert result["grounded"] is True
    assert "Based on the verified database" in result["response"]
