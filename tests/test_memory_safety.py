import sys
import tracemalloc
from typing import Any

from research_pipeline.rag.vector_db import VectorDB


def _get_process_rss() -> int | None:
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is not None:
        return psutil.Process().memory_info().rss

    try:
        import resource
    except ImportError:
        resource = None

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return int(usage.ru_maxrss)
        return int(usage.ru_maxrss) * 1024

    return None


def test_large_ingestion_memory_under_8gb():
    """Exercise real ingestion and verify memory stays far below 8 GiB."""
    db = VectorDB(chunk_size=50, chunk_overlap=10, max_embedding_batch_size=32)
    db.has_model = False
    db.model = None

    documents: list[dict[str, Any]] = []
    base_text = "AI medical research in healthcare clinical practice. " * 40
    for idx in range(3000):
        documents.append(
            {
                "title": f"Article {idx}",
                "text": base_text,
                "url": f"https://example.com/article-{idx}",
            }
        )

    before_rss = _get_process_rss()
    before_tracemalloc = None
    if before_rss is None:
        tracemalloc.start()

    db.index_documents(documents, retention_days=30)

    if before_rss is None:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        delta_bytes = peak
    else:
        after_rss = _get_process_rss()
        assert after_rss is not None
        delta_bytes = max(0, after_rss - before_rss)

    assert len(db.chunks) > 1000, "The ingestion path must create a large number of chunks."
    assert delta_bytes < 8 * 1024 ** 3, (
        f"Observed memory growth {delta_bytes} bytes exceeds 8 GiB during large ingestion."
    )
