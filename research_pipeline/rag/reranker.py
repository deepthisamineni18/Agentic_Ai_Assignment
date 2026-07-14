"""Re-ranking stage for the Conversational RAG Agent.

Spec requirement: "retrieves top-k relevant chunks ... re-ranks them using
cross-encoder or LLM-based re-ranking."

`CrossEncoderReranker` tries to load a real cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder`)
which jointly scores (query, passage) pairs -- this is strictly more accurate
than a bi-encoder/cosine-similarity first pass because it lets the model
attend across both texts instead of comparing independently-computed
embeddings. If the model can't be downloaded (offline environment, no
internet), it falls back to a BM25-style lexical re-ranker so the pipeline
never blocks or errors -- it just quietly loses the cross-attention benefit.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger("Reranker")

try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER_LIB = True
except ImportError:
    _HAS_CROSS_ENCODER_LIB = False

MODEL_LOAD_TIMEOUT_SECONDS = 15


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25Reranker:
    """Lexical fallback re-ranker (BM25) used when the cross-encoder model
    weights can't be downloaded. Not as strong as a real cross-encoder, but
    still a genuine second-pass re-rank over a different signal (term
    frequency / inverse document frequency / length normalization) than
    whatever produced the initial candidate list -- it is not a no-op."""

    K1 = 1.5
    B = 0.75

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_terms = _tokenize(query)
        doc_terms = [_tokenize(p) for p in passages]
        doc_lens = [len(d) for d in doc_terms]
        avg_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 1.0

        df = Counter()
        for d in doc_terms:
            for term in set(d):
                df[term] += 1
        n_docs = max(1, len(passages))

        scores = []
        for terms, dlen in zip(doc_terms, doc_lens):
            tf = Counter(terms)
            s = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                freq = tf[term]
                denom = freq + self.K1 * (1 - self.B + self.B * dlen / max(1.0, avg_len))
                s += idf * (freq * (self.K1 + 1)) / max(1e-9, denom)
            scores.append(s)
        return scores


class CrossEncoderReranker:
    """Re-ranks (query, chunk) pairs. Prefers a real cross-encoder; falls
    back to BM25 if the model can't be loaded (e.g. no internet access)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = None
        self.backend = "bm25_fallback"
        if _HAS_CROSS_ENCODER_LIB:
            try:
                # CrossEncoder's underlying huggingface_hub download call can
                # hang indefinitely (not just raise) when the network is
                # unreachable but not actively refusing connections (e.g. a
                # silently-dropping proxy/firewall) -- unlike a plain
                # ConnectionError, that case never reaches the except below
                # on its own. Bounding it in a thread with a hard timeout
                # guarantees this constructor always returns. Note:
                # shutdown(wait=False) is deliberate -- Python threads can't
                # be force-killed, so on timeout the load attempt keeps
                # running in the background and is simply abandoned; using
                # `with ThreadPoolExecutor()` here would defeat the timeout
                # entirely since its __exit__ blocks until the thread finishes.
                pool = ThreadPoolExecutor(max_workers=1)
                future = pool.submit(CrossEncoder, model_name)
                try:
                    self.model = future.result(timeout=MODEL_LOAD_TIMEOUT_SECONDS)
                    self.backend = "cross_encoder"
                finally:
                    pool.shutdown(wait=False)
            except FutureTimeoutError:
                logger.warning(
                    "Could not load CrossEncoder within %ss (network unreachable or too slow). "
                    "Falling back to BM25 re-ranking.", MODEL_LOAD_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logger.warning(
                    "Could not load CrossEncoder (%s). Falling back to BM25 re-ranking.", e
                )
        self._bm25 = _BM25Reranker()

    def rerank(self, query: str, candidates: list[tuple[float, object]], top_k: int) -> list[tuple[float, object]]:
        """`candidates` is a list of (initial_score, chunk) from the first-pass
        vector/lexical retrieval. Returns the top_k re-ranked as (new_score, chunk)."""
        if not candidates:
            return []

        passages = [c.text for _, c in candidates]
        chunks = [c for _, c in candidates]

        scored: list[tuple[float, object]] = []
        if self.backend == "cross_encoder":
            try:
                pairs = [[query, p] for p in passages]
                ce_scores = list(self.model.predict(pairs))
                if len(ce_scores) == len(chunks):
                    scored = list(zip(ce_scores, chunks))
            except Exception as e:
                logger.warning("Cross-encoder predict() failed at query time (%s); falling back to BM25 for this query.", e)

        if not scored:
            bm25_scores = self._bm25.score(query, passages)
            scored = list(zip(bm25_scores, chunks))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]
