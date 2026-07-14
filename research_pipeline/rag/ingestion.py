import datetime
import json
import logging
import os
import re

from research_pipeline.rag.search_tool import SearchTool
from research_pipeline.rag.vector_db import VectorDB

logger = logging.getLogger("IngestionPipeline")


def _estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\S+", text)))


def run_ingestion_pipeline(
    topic: str = "Advancements in AI in the medical field",
    db: VectorDB | None = None,
    output_dir: str = "output/ingestion_reports",
    max_sources: int = 10,
    retention_days: int = 30,
    token_budget: int | None = None,
) -> dict:
    """Executes the daily ingestion pipeline with graceful per-document handling."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting daily RAG ingestion for topic: '%s'", topic)

    searcher = SearchTool()
    collected = searcher.collect_documents(topic, max_sources=max_sources)

    if db is None:
        db = VectorDB()

    documents: list[dict[str, str]] = []
    processed_count = 0
    skipped_count = 0
    budget_hit = False
    current_tokens = 0

    for doc in collected:
        try:
            if not doc.title and not doc.text:
                raise ValueError("Document had neither title nor text")

            estimated_tokens = _estimate_tokens(doc.title) + _estimate_tokens(doc.text)
            if token_budget is not None and token_budget > 0:
                if current_tokens + estimated_tokens > token_budget and documents:
                    budget_hit = True
                    logger.info("Token budget reached; stopping ingestion before next document")
                    break
                if current_tokens + estimated_tokens > token_budget and not documents:
                    logger.info("Token budget reached for the first document; processing it anyway and stopping after this item")

            current_tokens += estimated_tokens
            documents.append({
                "text": doc.text,
                "title": doc.title,
                "url": doc.url,
                "url_status": getattr(doc, "url_status", None),
                "archived_url": getattr(doc, "archived_url", None),
                "published_days_ago": getattr(doc, "published_days_ago", None),
            })
            processed_count += 1
        except Exception as exc:  # pragma: no cover - defensive path
            skipped_count += 1
            logger.warning("Skipping document %s due to error: %s", doc.url if getattr(doc, "url", None) else "<unknown>", exc)

    indexing_stats = db.index_documents(documents, retention_days=retention_days)

    end_time = datetime.datetime.now(datetime.timezone.utc)
    duration = (end_time - start_time).total_seconds()

    if budget_hit:
        status = "PARTIAL"
    elif processed_count == 0:
        status = "FAILED"
    else:
        status = "SUCCESS"

    report = {
        "pipeline_run_id": f"ingest-{int(start_time.timestamp())}",
        "topic": topic,
        "execution_start": start_time.isoformat(),
        "execution_end": end_time.isoformat(),
        "duration_seconds": round(duration, 2),
        "articles_retrieved": len(collected),
        "articles_processed": processed_count,
        "articles_skipped": skipped_count,
        "token_budget": token_budget,
        "token_used_estimate": current_tokens,
        "new_chunks_added": indexing_stats["new_added"],
        "duplicates_skipped": indexing_stats["duplicate_count"],
        "total_chunks_in_db": indexing_stats["chunk_count"],
        "expired_chunks_purged": indexing_stats["purged"],
        "status": status,
    }

    report_file = os.path.join(output_dir, f"report_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(report_file, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Ingestion Report saved to %s", report_file)
    except Exception as exc:
        logger.error("Failed to write ingestion report: %s", exc)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion_pipeline()
