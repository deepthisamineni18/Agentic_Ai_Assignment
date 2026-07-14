import os
import json
import logging
import re
from research_pipeline.rag.vector_db import VectorDB
from research_pipeline.rag.reranker import CrossEncoderReranker
from research_pipeline.llm_client import LLMClient

logger = logging.getLogger("RAGAgent")

TOKEN_WINDOW_LIMIT = 500
GROUNDING_THRESHOLD = 0.30  # Stricter threshold based on sweep: better negative refusal with no recall loss in this analysis
RETRIEVAL_CANDIDATE_MULTIPLIER = 4  # over-fetch this many x top_k before re-ranking down
FOLLOW_UP_FILLER = {"tell", "more", "about", "second", "point", "that", "this", "it", "they"}
HIGH_ACCEPT_SCORE = 0.80  # require a high-confidence top score unless lexical/temporal checks rescue

# Phrases that indicate a query is a coreference/follow-up ("tell me more
# about that", "what about the second point") rather than a fresh, standalone
# question. IMPORTANT: these must be matched on word boundaries, not as raw
# substrings -- a plain `marker in text` check falsely matches short markers
# like "it" inside unrelated words (e.g. "capital" contains "it"), which was
# causing brand-new, unrelated questions to be misrouted through the
# follow-up path and grounded against the *previous* topic instead of
# actually being checked against the new question.
FOLLOW_UP_MARKERS = ["that", "they", "it", "this", "second point", "more about"]
_FOLLOW_UP_MARKER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in FOLLOW_UP_MARKERS) + r")\b"
)


def _has_follow_up_marker(text: str) -> bool:
    return bool(_FOLLOW_UP_MARKER_PATTERN.search(text.lower()))

RAG_SYSTEM_PROMPT = (
    "You are a research assistant answering questions using ONLY the numbered "
    "sources provided below. Rules:\n"
    "1. Only use information present in the sources. Never use outside knowledge.\n"
    "2. Cite every claim inline with its source number, e.g. [1], [2].\n"
    "3. If the sources don't fully answer the question, say what's missing rather than guessing.\n"
    "4. Be concise and directly answer the question first, then elaborate.\n"
    "5. If conversation history is provided, use it to resolve follow-up references "
    "(e.g. 'that', 'the second point'), but still ground every claim in the numbered sources."
)


class ConversationalRAGAgent:
    def __init__(self, vector_db: VectorDB, session_dir: str = "output/sessions",
                 reranker: CrossEncoderReranker | None = None, llm: LLMClient | None = None):
        self.vector_db = vector_db
        self.session_dir = session_dir
        self.reranker = reranker or CrossEncoderReranker()
        self.llm = llm or LLMClient()
        os.makedirs(self.session_dir, exist_ok=True)

    def _get_session_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir, f"session_{session_id}.json")

    def _load_history(self, session_id: str) -> list:
        path = self._get_session_path(session_id)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self, session_id: str, history: list) -> None:
        path = self._get_session_path(session_id)
        try:
            with open(path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session history: {e}")

    def _count_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def _summarize_history(self, history: list) -> list:
        """Summarizes older turns to manage token window context."""
        logger.info("Context window exceeded. Summarizing older turns...")
        if len(history) <= 2:
            return history

        retained = history[-2:]
        to_summarize = history[:-2]
        
        summary_points = []
        for turn in to_summarize:
            role = turn["role"].capitalize()
            content = turn["content"]
            summary_points.append(f"{role}: {content[:100]}...")
            
        summary_text = f"SUMMARY OF PAST TURNS: " + " | ".join(summary_points)
        summarized_history = [{"role": "system", "content": summary_text}] + retained
        
        return summarized_history

    def _strip_filler(self, text: str) -> str:
        return " ".join(
            w for w in text.split()
            if w.lower().strip(".,?!") not in FOLLOW_UP_FILLER
        )

    def answer_query(self, session_id: str, query: str) -> dict:
        history = self._load_history(session_id)
        
        # Enforce memory constraints
        total_tokens = sum(self._count_tokens(turn["content"]) for turn in history)
        if total_tokens > TOKEN_WINDOW_LIMIT:
            history = self._summarize_history(history)
            
        # Query expansion for coreference references to the last user turn.
        search_query = query
        grounding_query = query
        if history and _has_follow_up_marker(query):
            last_non_follow_up_user = next(
                (turn for turn in reversed(history)
                 if turn["role"] == "user" and not _has_follow_up_marker(turn["content"])),
                None,
            )
            preferred_user_turn = last_non_follow_up_user or next(
                (turn for turn in reversed(history) if turn["role"] == "user"), None
            )
            if preferred_user_turn:
                stripped_follow_up = self._strip_filler(query)
                grounding_query = preferred_user_turn["content"]
                search_query = f"{preferred_user_turn['content']} {stripped_follow_up}".strip()
                logger.info(
                    "Expanded follow-up query to: '%s' and grounding query to: '%s'",
                    search_query,
                    grounding_query,
                )

        temporal_constraint = self._extract_temporal_constraint(query)

        # Retrieve chunks (default k=5). Over-fetch a wider candidate pool
        # from the first-pass retriever, then re-rank down to top_k with the
        # cross-encoder (or its BM25 fallback) so the re-rank stage has
        # something meaningful to reorder rather than just relabeling the
        # same 5 items.
        #
        # Groundedness is judged from the *first-pass* retrieval score, not
        # the re-ranker's score: BM25/cross-encoder scores live on a
        # different, unbounded scale than GROUNDING_THRESHOLD was calibrated
        # against, so mixing them would make the grounding check meaningless.
        # The re-ranker only decides ordering among chunks that already
        # cleared the groundedness bar.
        top_k = 5
        candidate_pool = self.vector_db.query(
            grounding_query,
            top_k=top_k * RETRIEVAL_CANDIDATE_MULTIPLIER,
            max_age_days=temporal_constraint,
        )

        # Diagnostic logging: show first-pass retrieval summary
        try:
            logger.info("First-pass retrieval: %d candidates (top_k=%d, multiplier=%d)",
                        len(candidate_pool), top_k, RETRIEVAL_CANDIDATE_MULTIPLIER)
            if candidate_pool:
                for i, (score, chunk) in enumerate(candidate_pool[:3]):
                    logger.info(" candidate[%d] score=%.4f title=%r url=%r", i, float(score), chunk.metadata.get('title'), chunk.metadata.get('url'))
        except Exception:
            logger.debug("Failed to log candidate_pool details")

        if not candidate_pool or candidate_pool[0][0] < GROUNDING_THRESHOLD:
            raw_retrieved = []
        else:
            top_score, top_chunk = candidate_pool[0]
            # Choose acceptance threshold depending on reranker backend.
            # If the cross-encoder wasn't available and we fell back to BM25,
            # the re-ranker ordering lives on a different scale; accept any
            # first-pass score above `GROUNDING_THRESHOLD` in that case so
            # we don't reject valid corpus matches just because the heavy
            # reranker wasn't available.
            try:
                backend = getattr(self.reranker, "backend", None)
            except Exception:
                backend = None
            # Treat any non-cross-encoder backend as a BM25-like fallback;
            # this avoids hard rejection when the heavy cross-encoder model
            # isn't available or couldn't be loaded in time.
            if backend != "cross_encoder":
                accept_score = GROUNDING_THRESHOLD
            else:
                accept_score = HIGH_ACCEPT_SCORE
            logger.info(
                "Top-first-pass score=%.4f, using accept_score=%.2f (reranker backend=%r)",
                float(top_score),
                accept_score,
                backend,
            )
            # Allow temporal filtering to bypass strict lexical overlap when
            # the retrieved candidates were already constrained by recency.
            # This keeps recent/time-bound queries (e.g. "last week") from
            # being rejected just because they use temporal phrasing rather
            # than overlapping topical tokens.
            # Secondary sanity check: require a high top_score (e.g. 0.80)
            # unless lexical overlap or a temporal constraint rescues grounding.
            if top_score < accept_score and not (
                self._has_lexical_overlap(grounding_query, top_chunk.text)
                or temporal_constraint is not None
            ):
                raw_retrieved = []
            else:
                try:
                    raw_retrieved = self.reranker.rerank(search_query, candidate_pool, top_k=top_k)
                except Exception as e:
                    logger.warning("Re-ranker failed at query time (%s); treating as no re-rank result.", e)
                    raw_retrieved = []

        if not raw_retrieved:
            response = "I cannot answer this question because the retrieved database does not contain relevant, verified information on this topic."
            logger.info("Grounding check failed: no re-ranker candidates or top score below threshold for query %r", grounding_query)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
            self._save_history(session_id, history)
            return {"response": response, "sources": [], "grounded": False}

        # Generate grounded response
        citations_sources = []
        seen_urls = set()
        context_texts = []

        for score, chunk in raw_retrieved[:3]:
            url = chunk.metadata.get("url", "unknown")
            title = chunk.metadata.get("title", "Unknown Source")

            if url not in seen_urls:
                seen_urls.add(url)
                citations_sources.append({"id": len(citations_sources) + 1, "url": url, "title": title})

            cite_id = next(s["id"] for s in citations_sources if s["url"] == url)
            context_texts.append(f"Source [{cite_id}]: {chunk.text}")

        logger.info("Generating grounded answer using %d sources", len(citations_sources))
        answer_text = self._generate_answer(query, context_texts, history)

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer_text})
        self._save_history(session_id, history)

        return {"response": answer_text, "sources": citations_sources, "grounded": True}

    def _has_lexical_overlap(self, query: str, chunk_text: str) -> bool:
        query_terms = VectorDB._clean_terms(query)
        chunk_terms = VectorDB._clean_terms(chunk_text)
        return bool(query_terms & chunk_terms)
        

    def _extract_temporal_constraint(self, query: str) -> int | None:
        normalized = query.lower()
        if "last week" in normalized or "past week" in normalized or "this week" in normalized:
            return 7
        if "today" in normalized:
            return 1
        if "yesterday" in normalized:
            return 2
        if "last month" in normalized or "past month" in normalized or "this month" in normalized:
            return 30
        if "recent week" in normalized or "recent weeks" in normalized:
            return 7
        if "recent month" in normalized or "recent months" in normalized:
            return 30
        return None

    def _generate_answer(self, query: str, context_texts: list, history: list) -> str:
        """Generates the grounded answer text. Uses the real LLM
        (ANTHROPIC_API_KEY set) when available; falls back to the
        deterministic template otherwise, or if the LLM call fails for any
        reason (network, rate limit, etc.) — a live chat should never hard
        -fail just because the LLM backend hiccuped."""
        if self.llm.is_available():
            try:
                sources_block = "\n\n".join(context_texts)
                history_block = ""
                if history:
                    recent = history[-4:]
                    history_block = "\n\nRecent conversation:\n" + "\n".join(
                        f"{turn['role']}: {turn['content']}" for turn in recent
                    )
                user_prompt = f"Sources:\n{sources_block}{history_block}\n\nQuestion: {query}"
                return self.llm.generate(system=RAG_SYSTEM_PROMPT, user=user_prompt)
            except Exception as e:
                logger.warning("LLM generation failed (%s); falling back to template answer.", e)

        return (
            f"Based on the verified database, here is what I found:\n\n"
            + "\n\n".join(context_texts)
            + "\n\nThe above evidence is drawn directly from the indexed knowledge base. "
        )
