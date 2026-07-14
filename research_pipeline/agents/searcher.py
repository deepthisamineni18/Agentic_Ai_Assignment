from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

from research_pipeline.agents.base import BaseAgent
from research_pipeline.schemas import AgentMessage, ResearchPlan, SourceRecord

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.json"
logger = logging.getLogger("SearcherAgent")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


class _RateLimiter:
    """Simple token-bucket rate limiter shared across sub-query searches."""

    def __init__(self, max_calls_per_sec: float = 20.0):
        self.min_interval = 1.0 / max_calls_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class SearcherAgent(BaseAgent):
    inbox = "searcher"
    group = "searcher_group"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if CORPUS_PATH.exists():
            self._corpus = json.loads(CORPUS_PATH.read_text())
        else:
            logger.warning("Corpus file %s not found; generating a lightweight fallback corpus", CORPUS_PATH)
            from research_pipeline.data.generate_mock_corpus import generate
            self._corpus = generate(400)
            CORPUS_PATH.write_text(json.dumps(self._corpus))
        # pre-tokenize once for fast repeated scoring
        for rec in self._corpus:
            rec["_tokens"] = _tokenize(rec["title"] + " " + rec["snippet"])
        self._rate_limiter = _RateLimiter(max_calls_per_sec=25.0)

    def _mock_search(self, query_text: str, top_k: int) -> list[dict]:
        """Simulates a search API call against the pre-crawled corpus using
        Jaccard-similarity relevance scoring. Rate limited to emulate a real API."""
        self._rate_limiter.wait()
        q_tokens = _tokenize(query_text)
        scored = []
        for rec in self._corpus:
            overlap = q_tokens & rec["_tokens"]
            if not overlap:
                continue
            union = q_tokens | rec["_tokens"]
            score = len(overlap) / max(1, len(union))
            # Recency boost: newer articles score slightly higher
            recency_boost = max(0.0, (365 - min(rec["published_days_ago"], 365)) / 365) * 0.1
            scored.append((score + recency_boost, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def handle(self, message: AgentMessage) -> None:
        if message.msg_type != "plan.created":
            return
        plan = ResearchPlan.model_validate(message.payload)
        per_query_k = max(2, plan.max_sources // max(1, len(plan.sub_queries)) + 2)

        seen_urls: set[str] = set()
        collected: list[SourceRecord] = []

        # Respect Planner's strategy: breadth_first vs iterative_deepening
        strategy = getattr(plan, "strategy", "breadth_first") or "breadth_first"

        if strategy == "iterative_deepening":
            # Round-robin across sub-queries, taking top results per round,
            # which emulates iterative deepening where each angle is expanded
            # incrementally until `max_sources` are collected.
            per_round = max(1, per_query_k // 2)
            cursors = {sq.query_id: 0 for sq in plan.sub_queries}
            more = True
            while more and len(collected) < plan.max_sources:
                more = False
                for sq in plan.sub_queries:
                    results = self._mock_search(sq.query_text, top_k=per_query_k)
                    start = cursors[sq.query_id]
                    end = min(len(results), start + per_round)
                    for rec in results[start:end]:
                        if rec["url"] in seen_urls:
                            continue
                        seen_urls.add(rec["url"])
                        score = min(1.0, max(0.0, len(_tokenize(sq.query_text) & rec["_tokens"]) /
                                              max(1, len(_tokenize(sq.query_text)))))
                        collected.append(SourceRecord(
                            source_id=rec["source_id"],
                            url=rec["url"],
                            title=rec["title"],
                            snippet=rec["snippet"],
                            relevance_score=round(score, 3),
                            sub_query_id=sq.query_id,
                        ))
                        if len(collected) >= plan.max_sources:
                            break
                    cursors[sq.query_id] = end
                    if end < len(results):
                        more = True
                    if len(collected) >= plan.max_sources:
                        break
        else:
            # Default: breadth-first — exhaust each sub-query's top results in order
            for sq in plan.sub_queries:
                results = self._mock_search(sq.query_text, top_k=per_query_k)
                for rec in results:
                    if rec["url"] in seen_urls:
                        continue  # dedup across sub-queries
                    seen_urls.add(rec["url"])
                    score = min(1.0, max(0.0, len(_tokenize(sq.query_text) & rec["_tokens"]) /
                                          max(1, len(_tokenize(sq.query_text)))))
                    collected.append(SourceRecord(
                        source_id=rec["source_id"],
                        url=rec["url"],
                        title=rec["title"],
                        snippet=rec["snippet"],
                        relevance_score=round(score, 3),
                        sub_query_id=sq.query_id,
                        url_status={"status": None, "error": None},
                        archived_url=None,
                    ))
                if len(collected) >= plan.max_sources:
                    break

        collected.sort(key=lambda s: s.relevance_score, reverse=True)
        collected = collected[: plan.max_sources]

        self.logger.info(
            "Search complete for request=%s: %d unique sources from %d sub-queries",
            message.request_id, len(collected), len(plan.sub_queries),
        )
        self.emit(
            channel="synthesizer",
            request_id=message.request_id,
            recipient="SynthesizerAgent",
            msg_type="search.done",
            payload={
                "plan": plan.model_dump(),
                "sources": [s.model_dump() for s in collected],
            },
        )
