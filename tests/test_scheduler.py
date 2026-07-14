"""Unit tests for the CRON-style daily RAG ingestion scheduler."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Same offline-safe sentence_transformers stub used by test_rag.py, so
# VectorDB() doesn't attempt a real network call to huggingface.co here.
if "sentence_transformers" not in sys.modules or not isinstance(
    getattr(sys.modules.get("sentence_transformers"), "SentenceTransformer", None), type
):
    class _MockST:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name
        def encode(self, sentences, convert_to_numpy: bool = True):
            import numpy as np
            return np.ones((len(sentences), 384))
    sys.modules["sentence_transformers"] = MagicMock()
    import sentence_transformers
    sentence_transformers.SentenceTransformer = _MockST

from apscheduler.triggers.cron import CronTrigger
from pathlib import Path

from research_pipeline.rag.scheduler import RAGScheduler, _parse_cron


def test_parse_cron_valid_expression():
    trigger = _parse_cron("0 0 * * *")
    assert isinstance(trigger, CronTrigger)


def test_parse_cron_rejects_malformed_expression():
    import pytest
    with pytest.raises(ValueError):
        _parse_cron("not a cron expression")


def test_scheduler_run_once_executes_ingestion_immediately(tmp_path):
    """--run-once must fire the ingestion job synchronously without waiting
    for the cron trigger, and produce a report."""
    sched = RAGScheduler(topic="AI in medicine", cron_expr="0 0 * * *", output_dir=str(tmp_path))
    report = sched.run_once_now()
    assert report["status"] == "SUCCESS"
    assert report["topic"] == "AI in medicine"
    assert report["articles_retrieved"] > 0


def test_scheduler_start_registers_job_with_next_run_time(tmp_path):
    """start() should register a real cron job on the underlying
    APScheduler instance with a computed next_run_time (proves it's a true
    cron trigger, not a manual timer)."""
    sched = RAGScheduler(topic="AI in medicine", cron_expr="0 0 * * *", output_dir=str(tmp_path))
    sched.start()
    try:
        job = sched._scheduler.get_job("daily_rag_ingestion")
        assert job is not None
        assert job.next_run_time is not None
    finally:
        sched.shutdown()


def test_scheduler_persists_vector_db_across_runs(tmp_path):
    """The scheduler's VectorDB instance should accumulate chunks across
    successive scheduled runs rather than resetting each time."""
    sched = RAGScheduler(topic="AI in medicine", cron_expr="0 0 * * *", output_dir=str(tmp_path))
    sched.run_once_now()
    count_after_first = len(sched.db.chunks)
    sched.run_once_now()
    count_after_second = len(sched.db.chunks)
    assert count_after_second >= count_after_first


def test_scheduler_persists_vector_db_to_disk_across_process_restarts(tmp_path):
    """A new scheduler instance should load the same persisted VectorDB state."""
    sched = RAGScheduler(topic="AI in medicine", cron_expr="0 0 * * *", output_dir=str(tmp_path))
    report = sched.run_once_now()
    assert report["status"] == "SUCCESS"
    assert Path(sched.db_path).exists()

    count_after_first = len(sched.db.chunks)
    # Simulate restart by creating a fresh scheduler with the same output dir
    sched2 = RAGScheduler(topic="AI in medicine", cron_expr="0 0 * * *", output_dir=str(tmp_path))
    assert len(sched2.db.chunks) == count_after_first
    sched2.run_once_now()
    assert len(sched2.db.chunks) >= count_after_first
