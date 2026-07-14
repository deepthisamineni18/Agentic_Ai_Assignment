"""Unit tests for agent core logic (decomposition, scoring, section building)
that don't require a live Redis instance - pure function tests against each
agent's internal methods."""
from __future__ import annotations

import pytest

from research_pipeline.agents.critic import CriticAgent
from research_pipeline.agents.planner import PlannerAgent
from research_pipeline.agents.synthesizer import SynthesizerAgent
from research_pipeline.main import build_timing_breakdown, serialize_report
from research_pipeline.schemas import Depth, ReportSection, ResearchPlan, ResearchRequest, SourceRecord, SubQuery
from research_pipeline.supervisor import Supervisor


class _DummyBus:
    """Minimal stand-in so agents can be constructed without a real Redis connection."""
    pass


def make_planner() -> PlannerAgent:
    return PlannerAgent(bus=_DummyBus(), consumer_name="test")


def make_synthesizer() -> SynthesizerAgent:
    return SynthesizerAgent(bus=_DummyBus(), consumer_name="test")


def make_critic() -> CriticAgent:
    return CriticAgent(bus=_DummyBus(), consumer_name="test")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth,expected_n", [
    (Depth.SHALLOW, 3), (Depth.MODERATE, 5), (Depth.DEEP, 8),
])
def test_planner_subquery_count_matches_depth(depth, expected_n):
    planner = make_planner()
    sub_queries = planner._decompose("quantum computing", expected_n)
    assert len(sub_queries) == expected_n
    assert all(isinstance(sq, SubQuery) for sq in sub_queries)


def test_planner_subqueries_are_unique_and_reference_topic():
    planner = make_planner()
    sub_queries = planner._decompose("gene editing ethics", 5)
    texts = [sq.query_text for sq in sub_queries]
    assert len(set(texts)) == len(texts)  # all unique
    assert all("gene" in t and "editing" in t for t in texts)


def test_planner_strips_stopwords():
    planner = make_planner()
    sub_queries = planner._decompose("the impact of AI", 3)
    # "the", "of" should not appear as standalone tokens in the query core
    for sq in sub_queries:
        assert " the " not in f" {sq.query_text} "
        assert " of " not in f" {sq.query_text} "


def test_planner_uses_llm_decomposition_when_available(monkeypatch):
    planner = make_planner()

    class _FakeLLMClient:
        def __init__(self):
            self.calls = []

        def is_available(self):
            return True

        def generate(self, system, user, max_tokens=600, temperature=0.2):
            self.calls.append((system, user))
            return "Overview\nRecent developments\nPolicy implications"

    fake_llm = _FakeLLMClient()
    monkeypatch.setattr(planner, "_llm_client", fake_llm)

    sub_queries = planner._decompose("AI ethics", 3)

    assert len(sub_queries) == 3
    assert fake_llm.calls
    assert all(sq.query_text for sq in sub_queries)


def test_planner_falls_back_to_template_when_llm_call_fails(monkeypatch):
    planner = make_planner()

    class _FailingLLMClient:
        def is_available(self):
            return True

        def generate(self, system, user, max_tokens=600, temperature=0.2):
            raise RuntimeError("simulated API failure")

    monkeypatch.setattr(planner, "_llm_client", _FailingLLMClient())

    sub_queries = planner._decompose("AI ethics", 3)

    assert len(sub_queries) == 3
    assert all("ai" in sq.query_text.lower() or "ethics" in sq.query_text.lower() for sq in sub_queries)


def test_planner_falls_back_to_template_when_llm_unavailable(monkeypatch):
    planner = make_planner()

    class _UnavailableLLMClient:
        def is_available(self):
            return False

    monkeypatch.setattr(planner, "_llm_client", _UnavailableLLMClient())

    sub_queries = planner._decompose("AI ethics", 3)

    assert len(sub_queries) == 3
    assert all("ai" in sq.query_text.lower() or "ethics" in sq.query_text.lower() for sq in sub_queries)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def _plan(n=2) -> ResearchPlan:
    return ResearchPlan(
        request_id="r1",
        topic="test topic",
        strategy="breadth_first",
        max_sources=10,
        sub_queries=[SubQuery(query_id=f"q{i}", query_text=f"angle {i}", priority=i) for i in range(n)],
    )


def _source(sub_query_id, score=0.5, i=0) -> SourceRecord:
    return SourceRecord(
        source_id=f"s{i}",
        url=f"https://example{i % 3}.com/a",
        title="t",
        snippet="a useful snippet about the topic",
        relevance_score=score,
        sub_query_id=sub_query_id,
    )


def test_synthesizer_flags_gap_when_no_sources_for_subquery():
    synth = make_synthesizer()
    plan = _plan(n=2)
    sources = [_source("q0", 0.6, i=0)]  # nothing for q1
    sections = synth._build_sections(plan, sources)
    assert len(sections) == 2
    gap_section = [s for s in sections if s.citations == []][0]
    assert "gap" in gap_section.content.lower() or "no sufficiently relevant" in gap_section.content.lower()


def test_synthesizer_builds_citations_from_top_sources():
    synth = make_synthesizer()
    plan = _plan(n=1)
    sources = [_source("q0", 0.9, i=0), _source("q0", 0.8, i=1), _source("q0", 0.1, i=2)]
    sections = synth._build_sections(plan, sources)
    assert len(sections[0].citations) > 0
    assert set(sections[0].citations).issubset({"s0", "s1", "s2"})


def test_synthesizer_summary_mentions_gaps_when_present():
    synth = make_synthesizer()
    plan = _plan(n=2)
    sources = [_source("q0", 0.6, i=0)]
    sections = synth._build_sections(plan, sources)
    summary = synth._build_summary(plan.topic, sections, sources)
    assert "gap" in summary.lower()


def test_synthesizer_uses_llm_conflict_note_when_available(monkeypatch):
    synth = make_synthesizer()

    class _FakeLLMClient:
        def is_available(self):
            return True

        def generate(self, system, user, max_tokens=600, temperature=0.2):
            return "The sources disagree on framing and should be reconciled carefully."

    monkeypatch.setattr(synth, "_llm_client", _FakeLLMClient())

    plan = _plan(n=1)
    sources = [
        SourceRecord(source_id="s0", url="https://site-a.com/a", title="t", snippet="a", relevance_score=0.9, sub_query_id="q0"),
        SourceRecord(source_id="s1", url="https://site-b.com/a", title="t", snippet="b", relevance_score=0.8, sub_query_id="q0"),
        SourceRecord(source_id="s2", url="https://site-c.com/a", title="t", snippet="c", relevance_score=0.7, sub_query_id="q0"),
    ]
    sections = synth._build_sections(plan, sources)

    assert "reconciled carefully" in sections[0].content.lower()


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def test_critic_confidence_is_zero_for_no_sections():
    critic = make_critic()
    assert critic._score_confidence([], [], []) == 0.0


def test_critic_confidence_increases_with_coverage_and_relevance():
    critic = make_critic()
    good_sections = [ReportSection(heading="h", content="c", citations=["s0"])]
    good_sources = [_source("q0", 0.9, i=0)]
    bad_sections = [ReportSection(heading="h", content="c", citations=[])]
    bad_sources = [_source("q0", 0.05, i=0)]

    good_score = critic._score_confidence(good_sections, good_sources, gaps=[])
    bad_score = critic._score_confidence(bad_sections, bad_sources, gaps=["h"])
    assert good_score > bad_score


def test_critic_detects_domain_over_reliance_bias():
    critic = make_critic()
    sources = [
        SourceRecord(source_id=f"s{i}", url="https://samesite.com/a", title="t",
                     snippet="s", relevance_score=0.5, sub_query_id="q0")
        for i in range(5)
    ]
    flags = critic._detect_bias(sources)
    assert any("samesite.com" in f for f in flags)


def test_critic_no_bias_flags_for_diverse_high_relevance_sources():
    critic = make_critic()
    sources = [
        SourceRecord(source_id=f"s{i}", url=f"https://site{i}.com/a", title="t",
                     snippet="s", relevance_score=0.7, sub_query_id="q0")
        for i in range(5)
    ]
    flags = critic._detect_bias(sources)
    assert flags == []


def test_supervisor_can_run_without_redis():
    supervisor = Supervisor(redis_host="localhost", redis_port=6379)
    supervisor._use_inprocess = True
    request = ResearchRequest(topic="sample topic", depth=Depth.SHALLOW, max_sources=6, output_format="json")
    report = supervisor.run_single(request)
    assert report is not None
    assert report["topic"] == request.topic


def test_markdown_serialization_includes_sections_and_citations():
    report = {
        "report_id": "r1",
        "topic": "Test topic",
        "summary": "A short summary.",
        "sections": [
            {"heading": "Overview", "content": "A detailed summary.", "citations": ["s1"]}
        ],
        "sources": [{"source_id": "s1", "url": "https://example.com", "title": "Example", "relevance_score": 0.93, "scraped_at": "2026-01-01T00:00:00Z"}],
        "critique": {"confidence_score": 0.8, "gaps": [], "bias_flags": []},
        "metadata": {"timings": {"planning_time": 0.1, "search_time": 0.2, "scrape_time": 0.05, "synthesis_time": 0.15, "critique_time": 0.08, "re_search_time": 0.0}},
    }

    content = serialize_report(report, "markdown")
    assert "# Test topic" in content
    assert "## Overview" in content
    assert "https://example.com" in content


def test_timing_breakdown_uses_metadata_when_present():
    report = {"metadata": {"timings": {"planning_time": 0.1, "search_time": 0.2, "scrape_time": 0.05, "synthesis_time": 0.15, "critique_time": 0.08, "re_search_time": 0.0}}}
    breakdown = build_timing_breakdown(report)
    assert breakdown["planning_time"] == 0.1
    assert breakdown["re_search_time"] == 0.0
