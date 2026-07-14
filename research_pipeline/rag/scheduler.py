"""CRON-style scheduler for the daily Agentic RAG ingestion pipeline.

Satisfies the requirement: "The system operates as a scheduled pipeline
(CRON-style) that runs at the start of each day."

Uses APScheduler's ``CronTrigger`` (a real cron expression evaluator, not a
`time.sleep` loop) so the run time is configured exactly like a crontab
entry: minute/hour/day-of-week fields.

Usage
-----
Run continuously, firing every day at 00:00 UTC (default)::

    python -m research_pipeline.rag.scheduler --topic "Advancements in AI in the medical field"

Run continuously on a custom cron schedule (every day at 06:30 UTC)::

    python -m research_pipeline.rag.scheduler --cron "30 6 * * *"

Fire once immediately (useful for testing / CI) instead of waiting for the
next scheduled tick::

    python -m research_pipeline.rag.scheduler --run-once
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from research_pipeline.rag.ingestion import run_ingestion_pipeline
from research_pipeline.rag.vector_db import VectorDB

logger = logging.getLogger("RAGScheduler")


def _parse_cron(expr: str) -> CronTrigger:
    """Parses a standard 5-field crontab expression: `minute hour day month day_of_week`."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(
            f"Cron expression must have 5 fields (minute hour day month day_of_week), got: {expr!r}"
        )
    minute, hour, day, month, day_of_week = fields
    return CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week)


class RAGScheduler:
    """Owns a persistent VectorDB and fires `run_ingestion_pipeline` on a cron trigger."""

    def __init__(self, topic: str, cron_expr: str = "0 0 * * *", output_dir: str = "output/ingestion_reports"):
        self.topic = topic
        self.output_dir = output_dir
        self.db_path = str(Path(self.output_dir) / "vector_db.pkl")
        self.db = VectorDB(storage_path=self.db_path)
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._trigger = _parse_cron(cron_expr)
        self.cron_expr = cron_expr

    def _job(self) -> None:
        logger.info("Cron trigger fired — starting scheduled ingestion for topic=%r", self.topic)
        try:
            report = run_ingestion_pipeline(topic=self.topic, db=self.db, output_dir=self.output_dir)
            logger.info(
                "Scheduled ingestion complete: %d new chunks, %d duplicates skipped, %d total in DB",
                report["new_chunks_added"], report["duplicates_skipped"], report["total_chunks_in_db"],
            )
        except Exception:
            logger.exception("Scheduled ingestion run failed")

    def start(self) -> None:
        self._scheduler.add_job(self._job, self._trigger, id="daily_rag_ingestion", replace_existing=True)
        self._scheduler.start()
        next_run = self._scheduler.get_job("daily_rag_ingestion").next_run_time
        logger.info("RAG scheduler started. cron=%r  next_run=%s", self.cron_expr, next_run)

    def run_once_now(self) -> dict:
        """Fires the ingestion job immediately, bypassing the cron wait — for testing/CI."""
        logger.info("Running ingestion job once immediately (--run-once)")
        return run_ingestion_pipeline(topic=self.topic, db=self.db, output_dir=self.output_dir)

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("RAG scheduler stopped")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="CRON-style daily scheduler for the Agentic RAG ingestion pipeline")
    parser.add_argument("--topic", default="Advancements in AI in the medical field")
    parser.add_argument("--cron", default="0 0 * * *",
                         help="5-field crontab expression 'minute hour day month day_of_week'. Default: daily at 00:00 UTC.")
    parser.add_argument("--output-dir", default="output/ingestion_reports")
    parser.add_argument("--run-once", action="store_true",
                         help="Run the ingestion job immediately and exit, instead of waiting on the cron schedule.")
    args = parser.parse_args()

    sched = RAGScheduler(topic=args.topic, cron_expr=args.cron, output_dir=args.output_dir)

    if args.run_once:
        report = sched.run_once_now()
        print(f"Ingestion run complete: {report['new_chunks_added']} new chunks, "
              f"{report['duplicates_skipped']} duplicates skipped.")
        return

    sched.start()

    def _handle_sigterm(signum, frame):
        sched.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        sched.shutdown()


if __name__ == "__main__":
    main()
