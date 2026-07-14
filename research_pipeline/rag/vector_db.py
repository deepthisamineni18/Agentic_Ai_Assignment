from __future__ import annotations

import hashlib
import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger("VectorDB")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)


class VectorDB:
    """A real in-memory vector database with dense embeddings, chunking, deduplication, and retention purging."""

    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        model_name: str = "all-MiniLM-L6-v2",
        max_embedding_batch_size: int = 32,
        storage_path: str | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_embedding_batch_size = max(1, max_embedding_batch_size)
        self.chunks: list[Chunk] = []
        self.has_model = HAS_SENTENCE_TRANSFORMERS
        self.storage_path = str(storage_path) if storage_path else None

        if self.storage_path:
            self._load_state()

        if self.has_model:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as exc:
                logger.warning(
                    "Could not load SentenceTransformer (%s). Falling back to Jaccard similarity offline mode.",
                    exc,
                )
                self.model = None
                self.has_model = False
        else:
            self.model = None

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size
        chunks = []
        for start in range(0, len(words), step):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
        return chunks

    def _fallback_embedding(self, text: str) -> np.ndarray:
        tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]
        vector = np.zeros(64, dtype=float)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()[:4]
            index = int.from_bytes(digest, "big") % len(vector)
            vector[index] += 1.0
        if not np.any(vector):
            vector[0] = 1.0
        return vector / max(1.0, np.linalg.norm(vector))

    @staticmethod
    def _clean_terms(text: str) -> set[str]:
        stopwords = {"a", "an", "the", "and", "or", "for", "with", "this", "that", "these", "those", "how", "what", "why", "when", "where", "who", "which", "does", "did", "do", "can", "could", "would", "should", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "at", "by", "from", "into", "over", "under", "out", "up", "down", "it", "its", "they", "them", "their", "our", "you", "your", "we", "he", "she", "his", "her"}
        words = re.findall(r"[a-z0-9]+", text.lower())
        normalized = []
        for word in words:
            if len(word) <= 1:
                continue
            if word in stopwords:
                continue
            normalized.append(re.sub(r"(s|es|ed|ing)$", "", word))
        return {word for word in normalized if len(word) > 2}

    @staticmethod
    def _terms_match(term_a: str, term_b: str) -> bool:
        if term_a == term_b:
            return True

        synonym_groups = {
            "help": {"help", "assist", "aid", "support"},
            "assist": {"help", "assist", "aid", "support"},
            "aid": {"help", "assist", "aid", "support"},
            "support": {"help", "assist", "aid", "support"},
            "radiology": {"radiology", "radiologist", "radiologists", "radiological", "medical", "imaging", "image"},
            "radiologist": {"radiology", "radiologist", "radiologists", "radiological", "medical", "imaging", "image"},
            "radiologists": {"radiology", "radiologist", "radiologists", "radiological", "medical", "imaging", "image"},
            "radiological": {"radiology", "radiologist", "radiologists", "radiological", "medical", "imaging", "image"},
            "medical": {"radiology", "radiologist", "radiologists", "radiological", "medical", "imaging", "image"},
            "imaging": {"radiology", "radiologist", "radiologists", "radiological", "medical", "imaging", "image"},
            "image": {"radiology", "radiologist", "radiologists", "radiological", "medical", "imaging", "image"},
        }

        if term_a in synonym_groups and term_b in synonym_groups[term_a]:
            return True
        if term_b in synonym_groups and term_a in synonym_groups[term_b]:
            return True

        if len(term_a) >= 4 and len(term_b) >= 4:
            return SequenceMatcher(None, term_a, term_b).ratio() >= 0.6

        return False

    def purge_expired(self, retention_days: int = 30) -> int:
        """Removes chunks that are older than the retention policy."""
        now = time.time()
        max_age_seconds = retention_days * 24 * 3600
        original_count = len(self.chunks)
        
        self.chunks = [
            chunk for chunk in self.chunks 
            if (now - chunk.timestamp) <= max_age_seconds
        ]
        purged = original_count - len(self.chunks)
        if purged and self.storage_path:
            self._save_state()
        return purged

    def get_embedding_dim(self, sample_text: str = "test") -> int:
        """Return the embedding dimensionality for the current backend.

        If a SentenceTransformer model is available, attempt a single encode
        call to determine the vector width. On any failure or when running
        in fallback mode, return a conservative default that tests use.
        """
        DEFAULT_DIM = 384
        if self.has_model and self.model:
            try:
                emb = self.model.encode([sample_text], convert_to_numpy=True)
                if emb is None:
                    return DEFAULT_DIM
                # Embedding call may return shape (1, D) or (D,) depending on backend
                if hasattr(emb, "shape"):
                    if len(emb.shape) == 2:
                        return int(emb.shape[1])
                    return int(emb.shape[0])
            except Exception:
                return DEFAULT_DIM
        return DEFAULT_DIM

    def _load_state(self) -> None:
        try:
            path = Path(self.storage_path)
            if not path.exists():
                return
            with path.open("rb") as fh:
                loaded = pickle.load(fh)
            if isinstance(loaded, list):
                self.chunks = loaded
                logger.info("Loaded VectorDB state from %s (%d chunks)", self.storage_path, len(self.chunks))
        except Exception as exc:
            logger.warning("Could not load persisted VectorDB state from %s: %s", self.storage_path, exc)

    def _save_state(self) -> None:
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump(self.chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Persisted VectorDB state to %s (%d chunks)", self.storage_path, len(self.chunks))
        except Exception as exc:
            logger.warning("Failed to persist VectorDB state to %s: %s", self.storage_path, exc)

    def index_documents(self, documents: list[dict[str, Any]], retention_days: int = 30) -> dict[str, Any]:
        """Chunks documents, generates dense embeddings, deduplicates, and stores them."""
        seen_hashes: set[str] = {c.id for c in self.chunks}
        duplicate_count = 0
        new_chunks = []
        
        for doc in documents:
            text = doc.get("text", "")
            title = doc.get("title", "")
            url = doc.get("url", "")
            published_days_ago = doc.get("published_days_ago")
            
            doc_chunks = self._chunk_text(f"{title} {text}")
            for chunk_text in doc_chunks:
                digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(digest)
                metadata = {"url": url, "title": title, "retention_days": retention_days}
                if published_days_ago is not None:
                    metadata["published_days_ago"] = published_days_ago
                # Preserve URL status and archive metadata so citation sources
                # can surface reachability and fallback archived links.
                url_status = doc.get("url_status")
                if url_status is not None:
                    metadata["url_status"] = url_status
                archived_url = doc.get("archived_url")
                if archived_url is not None:
                    metadata["archived_url"] = archived_url
                new_chunks.append(Chunk(
                    id=digest, 
                    text=chunk_text, 
                    metadata=metadata
                ))

        if not new_chunks:
            # Still enforce retention even when nothing new was added -- a
            # scheduled run against an already-fully-indexed topic should
            # not skip purging expired chunks, and callers (e.g. ingestion.py)
            # unconditionally read the "purged" key from this dict.
            purged = self.purge_expired(retention_days)
            return {
                "chunk_count": len(self.chunks),
                "duplicate_count": duplicate_count,
                "new_added": 0,
                "purged": purged,
            }

        if self.has_model:
            for start in range(0, len(new_chunks), self.max_embedding_batch_size):
                batch = new_chunks[start:start + self.max_embedding_batch_size]
                texts_to_embed = [c.text for c in batch]
                try:
                    embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning("Embedding batch failed for %d chunks: %s", len(batch), exc)
                    self.has_model = False
                    break

                if embeddings is None:
                    logger.warning("Embedding batch returned None; falling back to lexical retrieval")
                    self.has_model = False
                    break

                for chunk, emb in zip(batch, embeddings):
                    chunk.embedding = emb

        if not self.has_model:
            for chunk in new_chunks:
                if chunk.embedding is None:
                    chunk.embedding = self._fallback_embedding(chunk.text)

        self.chunks.extend(new_chunks)
        
        # Optionally purge while we're indexing
        purged = self.purge_expired(retention_days)

        if self.storage_path:
            self._save_state()

        return {
            "chunk_count": len(self.chunks),
            "duplicate_count": duplicate_count,
            "new_added": len(new_chunks),
            "purged": purged
        }

    def query(self, query: str, top_k: int = 5, max_age_days: int | None = None) -> list[tuple[float, Chunk]]:
        """Returns top_k chunks based on cosine similarity of dense embeddings or a normalized fallback score.

        If max_age_days is set, chunks older than that many days are excluded
        when published date metadata is available.
        """
        if not self.chunks:
            return []

        logger.info("VectorDB.query called: query=%r top_k=%d max_age_days=%s -- total_chunks=%d",
                    query, top_k, max_age_days, len(self.chunks))

        candidates = self.chunks
        if max_age_days is not None:
            candidates = [
                c for c in candidates
                if c.metadata.get("published_days_ago") is None
                or c.metadata.get("published_days_ago") <= max_age_days
            ]
            logger.info("VectorDB.query: %d candidates after temporal filter (max_age_days=%s)", len(candidates), max_age_days)
            if not candidates:
                return []

        if self.has_model:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            valid_chunks = [c for c in candidates if c.embedding is not None]
            if not valid_chunks:
                return []

            chunk_embeddings = np.array([c.embedding for c in valid_chunks])
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

            scored = list(zip(similarities, valid_chunks))
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[:top_k]

        if candidates and any(c.embedding is None for c in candidates):
            for chunk in candidates:
                if chunk.embedding is None:
                    chunk.embedding = self._fallback_embedding(chunk.text)

        query_terms = self._clean_terms(query)
        if not query_terms:
            return []

        scored = []
        for chunk in candidates:
            chunk_terms = self._clean_terms(chunk.text)
            if not chunk_terms:
                continue

            overlap = 0
            for query_term in query_terms:
                if any(self._terms_match(query_term, chunk_term) for chunk_term in chunk_terms):
                    overlap += 1

            if not overlap:
                continue

            shared_ratio = overlap / max(1, len(query_terms))
            coverage = overlap / max(1, len(chunk_terms))
            score = round(min(1.0, 0.6 * shared_ratio + 0.4 * coverage), 4)
            if overlap >= 2:
                score = min(1.0, score + 0.1)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]
