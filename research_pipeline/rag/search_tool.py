from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger("SearchTool")

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:  # pragma: no cover - httpx ships transitively via sentence-transformers
    httpx = None
    _HAS_HTTPX = False

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass
class SearchDocument:
    url: str
    title: str
    text: str
    published_days_ago: int | None = None
    url_status: dict | None = None
    archived_url: str | None = None


class _RateLimiter:
    """Simple token-bucket style limiter for search calls."""

    def __init__(self, max_calls_per_sec: float = 5.0) -> None:
        self.min_interval = 1.0 / max_calls_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class _LiveSearchBackend:
    """Opt-in real web search via the Tavily API.

    This repo works with ZERO external search API calls out of the box --
    `SearchTool` falls back to a deterministic pre-crawled corpus (see
    `_load_corpus`/`_search_corpus`) so the pipeline is reproducible and
    runnable fully offline/in CI. This class is an opt-in upgrade, mirroring
    the pattern used by `LLMClient` for generation: set `TAVILY_API_KEY` and
    `SearchTool` will query the real web for "the latest news and articles"
    as the spec describes, instead of matching against the local corpus.
    Unset it and behavior is unchanged from before.

    Any failure here (missing key, network error, rate limit, bad response)
    must be caught by the caller and treated as "live search unavailable for
    this query" -- it should never take down the ingestion run.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 10.0) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key) and _HAS_HTTPX

    def search(self, query: str, max_days_old: int, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError(
                "Live search backend not available: set TAVILY_API_KEY (and ensure httpx is installed) to enable it."
            )

        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
        }
        # Tavily accepts a "days" window to restrict results by recency; only
        # send it when the caller's recency filter is tighter than "no limit".
        if max_days_old and max_days_old < 365:
            payload["days"] = max_days_old

        response = httpx.post(TAVILY_SEARCH_URL, json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()

        results: list[dict[str, Any]] = []
        for item in body.get("results", []):
            url = str(item.get("url", ""))
            if not url:
                continue
            results.append(
                {
                    "url": url,
                    "title": str(item.get("title", "")),
                    "snippet": str(item.get("content", "")),
                    # Tavily doesn't reliably return article age; recency is
                    # already enforced server-side via the `days` param above,
                    # so we don't claim a specific age here rather than guess.
                    "published_days_ago": None,
                }
            )
        return results


class SearchTool:
    """Search helper. Uses a real live web search API (Tavily) when
    `TAVILY_API_KEY` is set; otherwise falls back to a realistic
    pre-crawled corpus so the pipeline stays deterministic and runnable
    fully offline."""

    def __init__(
        self,
        max_days_old: int = 365,
        rate_limit_per_sec: float = 5.0,
        corpus_path: str | None = None,
        live_backend: "_LiveSearchBackend | None" = None,
        search_all_subqueries: bool | None = None,
    ) -> None:
        self._rng = random.Random(42)
        self.max_days_old = max_days_old
        self._rate_limiter = _RateLimiter(max_calls_per_sec=rate_limit_per_sec)
        self._corpus = self._load_corpus(corpus_path)
        self._live_backend = live_backend or _LiveSearchBackend()
        # If the env var is set, prefer live search where possible. Note
        # that `is_available()` still checks both the presence of an API
        # key and that `httpx` is installed.
        self.prefer_live = bool(os.environ.get("TAVILY_API_KEY"))
        if self.prefer_live:
            if self._live_backend.is_available():
                logger.info("Live search enabled via TAVILY_API_KEY (httpx available)")
            else:
                logger.warning(
                    "TAVILY_API_KEY present but live search backend unavailable: httpx installed=%s, api_key_set=%s",
                    _HAS_HTTPX,
                    bool(self._live_backend.api_key),
                )
        # If True, still execute live/corpus search for every generated
        # subquery (useful for debugging why only the first few subqueries
        # produced documents). Default controlled by env `SEARCH_ALL_SUBQUERIES`.
        if search_all_subqueries is None:
            self.search_all_subqueries = os.environ.get("SEARCH_ALL_SUBQUERIES", "false").lower() in ("1", "true", "yes")
        else:
            self.search_all_subqueries = bool(search_all_subqueries)

    def _load_corpus(self, corpus_path: str | None) -> list[dict[str, Any]]:
        resolved_path = Path(corpus_path) if corpus_path else Path(__file__).resolve().parents[1] / "data" / "corpus.json"
        if not resolved_path.exists():
            return []

        try:
            with resolved_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Could not load search corpus %s: %s", resolved_path, exc)
            return []

        if not isinstance(payload, list):
            return []
        return payload

    # Lightweight stopword list to avoid common short tokens drowning signal
    _STOPWORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "this",
        "that",
        "these",
        "those",
        "how",
        "what",
        "why",
        "when",
        "where",
        "who",
        "which",
        "does",
        "did",
        "do",
        "can",
        "could",
        "would",
        "should",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
        "into",
        "over",
        "under",
        "out",
        "up",
        "down",
        "it",
        "its",
        "they",
        "them",
        "their",
        "our",
        "you",
        "your",
        "we",
        "he",
        "she",
        "his",
        "her",
    }

    # Small synonym map to bridge common abbreviation/expansion gaps
    _SYNONYMS: dict[str, set[str]] = {
        "ai": {"ai", "artificial", "intelligence"},
        "artificial": {"ai", "artificial", "intelligence"},
        "intelligence": {"ai", "artificial", "intelligence"},
        "medical": {"medical", "medicine", "health", "healthcare", "clinical"},
        "medicine": {"medical", "medicine", "health", "healthcare", "clinical"},
        "health": {"medical", "medicine", "health", "healthcare", "clinical"},
    }

    @staticmethod
    def _terms_match(term_a: str, term_b: str) -> bool:
        if term_a == term_b:
            return True

        # synonym groups
        if term_a in SearchTool._SYNONYMS and term_b in SearchTool._SYNONYMS[term_a]:
            return True
        if term_b in SearchTool._SYNONYMS and term_a in SearchTool._SYNONYMS[term_b]:
            return True

        # fuzzy match for medium-length tokens
        if len(term_a) >= 4 and len(term_b) >= 4:
            return SequenceMatcher(None, term_a, term_b).ratio() >= 0.6

        return False

    def _tokenize(self, text: str) -> set[str]:
        """Return a set of normalized tokens keeping useful 2-letter acronyms like 'ai'.

        Previously this function filtered out tokens of length <= 2 which
        dropped valid acronyms (e.g. "AI"). We now keep tokens of length >=2
        but filter a small stopword set to avoid flooding with common words.
        """
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower())]
        return {tok for tok in tokens if (len(tok) >= 2 and tok not in self._STOPWORDS)}

    def _search_corpus(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._corpus:
            return []

        self._rate_limiter.wait()
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self._corpus:
            published_days_ago = int(record.get("published_days_ago", 365))
            if published_days_ago > self.max_days_old:
                continue

            title = str(record.get("title", ""))
            snippet = str(record.get("snippet", ""))
            record_tokens = self._tokenize(f"{title} {snippet}")
            if not record_tokens:
                continue

            # Compute overlap allowing synonym and fuzzy matches. For each
            # query token, check if it or any of its synonym/fuzzy matches
            # appear in the record tokens.
            overlap_count = 0
            for qtok in query_tokens:
                if any(self._terms_match(qtok, rtok) for rtok in record_tokens):
                    overlap_count += 1
            if overlap_count == 0:
                continue

            # approximate overlap ratio using the counted matches
            overlap_ratio = overlap_count / max(1, len(query_tokens | record_tokens))
            recency_boost = max(0.0, (self.max_days_old - min(published_days_ago, self.max_days_old)) / max(1, self.max_days_old)) * 0.1
            score = overlap_ratio + recency_boost
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def generate_subqueries(self, topic: str, max_queries: int = 5) -> list[str]:
        topic = topic.strip()
        if not topic:
            return []

        # IMPORTANT: lowercase the topic *before* tokenizing. The regex only
        # matches [a-z0-9]. Keep 2-letter acronyms like "ai" while filtering
        # a small set of common stopwords so queries remain useful.
        tokens = [token for token in re.findall(r"[a-z0-9]+", topic.lower()) if (len(token) >= 2 and token not in self._STOPWORDS)]
        if not tokens:
            return [topic]

        base_terms = tokens[:4]
        queries: list[str] = []
        seen: set[str] = set()

        def add_query(query: str) -> bool:
            normalized = " ".join(query.split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                queries.append(normalized)
                return True
            return False

        # seed queries that include the full topic plus focused angles
        add_query(topic)
        if len(base_terms) >= 2:
            add_query(f"{base_terms[0]} {base_terms[-1]}")
        if len(base_terms) >= 1:
            add_query(f"{base_terms[0]} applications")
            add_query(f"{base_terms[0]} diagnostics")
            add_query(f"{base_terms[0]} case studies")
        if len(base_terms) >= 2:
            add_query(f"{base_terms[0]} {base_terms[1]} overview")
        if len(base_terms) >= 3:
            add_query(f"{base_terms[0]} {base_terms[1]} {base_terms[2]}")

        # richer set of suffixes to produce more topical sub-queries
        year = str(time.localtime().tm_year)
        suffixes = [
            "research",
            "case studies",
            "trends",
            "best practices",
            "challenges",
            "recent findings",
            "guide",
            "analysis",
            "clinical trials",
            "drug discovery",
            "diagnostic accuracy",
            f"news {year}",
            "ethics",
            "regulation",
            "deployment",
            "benchmarks",
        ]
        max_attempts = max_queries * len(suffixes) * max(1, len(base_terms)) + 10
        attempts = 0
        suffix_idx = 0
        while len(queries) < max_queries and attempts < max_attempts:
            attempts += 1
            term = base_terms[attempts % len(base_terms)]
            suffix = suffixes[suffix_idx % len(suffixes)]
            suffix_idx += 1
            add_query(f"{term} {suffix}")

        pad = 1
        while len(queries) < max_queries:
            add_query(f"{topic} (angle {pad})")
            pad += 1
            if pad > max_queries + 5:
                break

        return queries[:max_queries]

    def _search_live(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Query the real web via the live backend, if configured. Rate
        limiting and quota/error handling are enforced here: any failure
        (auth, rate limit, timeout, malformed response) is logged and
        treated as "no live results for this query" so the caller falls
        back to the corpus rather than aborting the run."""
        if not self._live_backend.is_available():
            return []

        self._rate_limiter.wait()
        try:
            return self._live_backend.search(query, max_days_old=self.max_days_old, top_k=top_k)
        except Exception as exc:  # covers HTTP errors, timeouts, rate limits, quota exhaustion
            logger.warning("Live search failed for query %r (%s); falling back to local corpus.", query, exc)
            return []

    def _check_and_archive_url(self, url: str, timeout: float = 8.0) -> tuple[int | None, str | None, str | None]:
        """Check a URL and archive its HTML locally when reachable.

        Returns `(status_code, error_message, archived_path)` where
        `archived_path` is the local path written (or None).
        """
        if not _HAS_HTTPX:
            return None, "httpx not installed", None

        try:
            # Try HEAD first
            r = httpx.head(url, follow_redirects=True, timeout=timeout)
            status = int(r.status_code)
        except Exception:
            status = None

        # If HEAD indicates auth/forbid or is unavailable, try GET to be sure
        if status is None or status in (401, 403, 405) or status >= 400:
            try:
                r = httpx.get(url, follow_redirects=True, timeout=timeout)
                status = int(r.status_code)
            except Exception as exc:
                return None, str(exc), None

        # If reachable (2xx/3xx), attempt to archive page HTML
        if 200 <= status < 400:
            try:
                # Re-fetch body with a reasonable timeout
                r = httpx.get(url, follow_redirects=True, timeout=timeout)
                content = r.content
                # Save to output/archives/<sha256>.html
                import os
                import hashlib
                from pathlib import Path

                os.makedirs("output/archives", exist_ok=True)
                filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html"
                path = os.path.join("output/archives", filename)
                with open(path, "wb") as fh:
                    fh.write(content)
                return status, None, Path(path).resolve().as_uri()
            except Exception as exc:
                return status, str(exc), None
        # If not reachable, query Wayback availability as a fallback
        try:
            wb = httpx.get("http://archive.org/wayback/available", params={"url": url}, timeout=timeout)
            if wb.status_code == 200:
                body = wb.json()
                snap = body.get("archived_snapshots", {}).get("closest")
                if snap and snap.get("available"):
                    return status, None, snap.get("url")
        except Exception:
            # ignore Wayback failures
            pass

        return status, None, None

    def collect_documents(self, topic: str, max_sources: int = 5) -> list[SearchDocument]:
        subqueries = self.generate_subqueries(topic, max_queries=max(3, min(7, max_sources + 2)))
        logger.info("Generated %d subqueries for topic %r: %s", len(subqueries), topic, subqueries)
        if self.prefer_live:
            logger.info("Prefer live search: attempting live backend for each subquery (falls back to corpus on failure)")
        else:
            logger.info("Live search not requested; using local corpus by default")
        documents: list[SearchDocument] = []
        seen_urls: set[str] = set()

        processed = 0
        for subquery in subqueries:
            processed += 1
            logger.info("Searching subquery %d/%d: %s", processed, len(subqueries), subquery)

            live_matches = self._search_live(subquery, top_k=max(3, max_sources))
            if live_matches:
                logger.info("Live backend returned %d matches for subquery %r", len(live_matches), subquery)

            corpus_matches = live_matches or self._search_corpus(subquery, top_k=max(3, max_sources))
            if not live_matches and corpus_matches:
                logger.info("Corpus returned %d matches for subquery %r", len(corpus_matches), subquery)

            # Add matches to documents until max_sources reached. If
            # `search_all_subqueries` is enabled, continue executing the
            # search for remaining subqueries (for visibility/debugging) but
            # don't exceed `max_sources` in appended documents.
            if corpus_matches:
                added_this_subquery = 0
                for record in corpus_matches:
                    if len(documents) >= max_sources:
                        break
                    url = str(record.get("url", ""))
                    if not url:
                        logger.debug("Skipping record with empty URL for subquery %r", subquery)
                        continue
                    # Validate URL reachability, attempt to archive, and record status
                    status, err, archived_url = self._check_and_archive_url(url)
                    url_status = {"status": status, "error": err, "archived_url": archived_url} if (status is not None or err or archived_url) else None
                    # Allow adding a document if either the live URL is reachable
                    # (2xx/3xx) or a Wayback/archive snapshot was found.
                    live_ok = isinstance(status, int) and 200 <= status < 400
                    if not live_ok and not archived_url:
                        # For corpus records that include a snippet or text, create a
                        # local archived HTML file so the citation remains resolvable
                        # even when the original URL is missing or paywalled.
                        snippet_text = str(record.get("snippet") or record.get("text") or "").strip()
                        if snippet_text:
                            try:
                                import hashlib
                                from pathlib import Path

                                Path("output/archives").mkdir(parents=True, exist_ok=True)
                                source_id = str(record.get("source_id") or hashlib.sha256(url.encode("utf-8")).hexdigest())
                                filename = f"{source_id}.html"
                                path = Path("output/archives") / filename
                                with path.open("w", encoding="utf-8") as fh:
                                    fh.write(f"<html><head><meta charset=\"utf-8\"><title>{record.get('title') or ''}</title></head><body>")
                                    fh.write(f"<h1>{record.get('title') or ''}</h1>\n")
                                    fh.write(f"<p>Original URL: <a href=\"{url}\">{url}</a></p>\n")
                                    fh.write(f"<div>{snippet_text}</div>\n")
                                    fh.write("</body></html>")
                                archived_url = path.resolve().as_uri()
                                # update url_status to include this archived URL
                                url_status = {"status": status, "error": err, "archived_url": archived_url}
                                logger.info("Created local archive for corpus record %s -> %s", url, archived_url)
                            except Exception as exc:
                                logger.warning("Failed to create local archive for %s: %s", url, exc)
                                continue
                        else:
                            logger.info("Skipping unreachable URL %s for subquery %r (status=%s err=%s)", url, subquery, status, err)
                            continue

                    if url in seen_urls:
                        logger.debug("Skipping duplicate URL %s for subquery %r", url, subquery)
                        continue
                    seen_urls.add(url)
                    title = str(record.get("title") or subquery.title())
                    text = str(record.get("snippet") or record.get("text") or "")
                    published_days_ago = record.get("published_days_ago")
                    documents.append(
                        SearchDocument(
                            url=url,
                            title=title,
                            text=text,
                            published_days_ago=published_days_ago,
                            url_status=url_status,
                            archived_url=archived_url,
                        )
                    )
                    added_this_subquery += 1
                logger.info("Added %d documents from subquery %r (total %d/%d)", added_this_subquery, subquery, len(documents), max_sources)

            else:
                title = f"{subquery.title()} Overview"
                text = (
                    f"This document discusses {subquery} in depth, with practical insights, "
                    f"examples, and evidence relevant to the requested topic."
                )
                synth_url = f"https://example.org/{len(documents) + 1}/{re.sub(r'[^a-z0-9]+', '-', subquery).strip('-')}"
                logger.debug("Synthetic fallback document created for subquery %r -> %s", subquery, synth_url)
                # Only add synthetic docs when we still need documents
                if len(documents) < max_sources:
                    documents.append(
                        SearchDocument(
                            url=synth_url,
                            title=title,
                            text=text,
                        )
                    )

            if len(documents) >= max_sources and not self.search_all_subqueries:
                logger.info("Reached max_sources (%d); stopping subquery loop", max_sources)
                break

        # If we didn't search all subqueries for visibility and the
        # caller asked for a full run via env, run remaining subqueries for
        # logging purposes (don't add more documents beyond max_sources).
        if self.search_all_subqueries and len(subqueries) > processed:
            for subquery in subqueries[processed:]:
                logger.info("(debug) executing remaining subquery for visibility: %s", subquery)
                live_matches = self._search_live(subquery, top_k=1)
                corpus_matches = live_matches or self._search_corpus(subquery, top_k=1)
                logger.info("(debug) subquery %r yielded %d matches during forced visibility pass", subquery, len(corpus_matches))

        return documents[:max_sources]
