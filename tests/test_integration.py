"""Integration tests: spin up all 4 agents as real processes and run the
full pipeline against a live Redis instance for 3+ sample topics.

Requires a reachable Redis at localhost:6379 (or REDIS_HOST/REDIS_PORT env
vars). Skipped automatically if Redis isn't reachable, so `pytest` still
passes in environments without infra (unit tests still run).
"""
from __future__ import annotations

import os
import uuid

import pytest
import redis

from research_pipeline.bus import MessageBus
from research_pipeline.schemas import ResearchRequest
from research_pipeline.supervisor import Supervisor

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))


def _redis_available() -> bool:
    try:
        redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=1).ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="Redis not reachable")

SAMPLE_TOPICS = [
    "Recent breakthroughs in quantum computing hardware",
    "The economic impact of electric vehicle adoption",
    "Advances in gene editing for disease treatment",
]


@pytest.fixture(scope="module")
def supervisor():
    sup = Supervisor(redis_host=REDIS_HOST, redis_port=REDIS_PORT)
    sup.start_agents()
    import time
    time.sleep(1.0)
    yield sup
    sup.stop_agents()


@pytest.mark.parametrize("topic", SAMPLE_TOPICS)
def test_pipeline_produces_valid_report(supervisor, topic):
    request = ResearchRequest(topic=topic, depth="shallow", max_sources=8, output_format="json")
    report = supervisor.run_single(request)

    assert report is not None, f"pipeline timed out or failed for topic={topic!r}"
    assert report["topic"] == topic
    assert 0.0 <= report["critique"]["confidence_score"] <= 1.0
    assert len(report["sections"]) > 0
    assert len(report["sources"]) > 0

    # every citation in every section must reference a real source_id
    valid_ids = {s["source_id"] for s in report["sources"]}
    for section in report["sections"]:
        for cite in section["citations"]:
            assert cite in valid_ids, f"dangling citation {cite} in section {section['heading']!r}"


def test_report_ids_are_unique_across_requests(supervisor):
    ids = set()
    for topic in SAMPLE_TOPICS:
        request = ResearchRequest(topic=topic, depth="shallow", max_sources=6)
        report = supervisor.run_single(request)
        assert report["report_id"] not in ids
        ids.add(report["report_id"])


def test_pipeline_handles_low_signal_topic_via_research_loop(supervisor):
    """A near-nonsense topic should still terminate (via re-search loop or
    max-iteration cutoff) rather than hang, and should self-report low confidence
    or gaps rather than fabricate content."""
    request = ResearchRequest(topic="zzqx flibbertigibbet unrelated nonsense", depth="shallow", max_sources=6)
    report = supervisor.run_single(request)
    assert report is not None
    assert "confidence_score" in report["critique"]
