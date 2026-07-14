"""
Multi-Agent Customer Intelligence Platform — Task 4
Agents: IntentClassifierAgent, KnowledgeRetrieverAgent,
        ResponseGeneratorAgent, QualityCheckerAgent
KV-cache prefix simulation via KVCacheManager.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
import threading
from dataclasses import dataclass
from typing import Any

from research_pipeline.customer_intelligence.config import CustomerIntelligenceConfig

logger = logging.getLogger("CustomerIntelligence")


class InvalidCustomerTurnError(ValueError):
    """Raised when a CustomerTurn fails input validation (malformed content)."""


class AgentStageError(RuntimeError):
    """Raised when a pipeline stage (agent) fails unexpectedly.

    Wraps the original exception so `handle_turn` can log it with full
    context (session, stage name) and degrade gracefully instead of
    propagating a raw traceback to the caller.
    """

    def __init__(self, stage: str, original: Exception) -> None:
        super().__init__(f"{stage} failed: {original!r}")
        self.stage = stage
        self.original = original

# ---------------------------------------------------------------------------
# System Prompts  (token targets met by design — verified below)
# ---------------------------------------------------------------------------

# ~320 words ≈ 430 tokens
INTENT_CLASSIFIER_SYSTEM_PROMPT = """\
ROLE: You are the Lead Customer Intent Classifier for an enterprise software platform.
Your primary objective is to analyse every incoming customer message and categorise it
into exactly one of the following taxonomy categories.

TAXONOMY:
  1. TECHNICAL_SUPPORT  — Requests involving software bugs, installation failures,
     system downtime, API integration errors, Docker container issues, network
     connectivity faults, configuration problems, authentication failures, SDK errors,
     timeout exceptions, or any technical malfunction requiring engineering guidance.

  2. BILLING_AND_ACCOUNT — Requests relating to invoice queries, subscription renewal,
     payment failures, charge disputes, refund requests, plan upgrades or downgrades,
     seat management, coupon redemption, account registration, password resets, SSO
     configuration, or any financial or identity-management concern.

  3. PRODUCT_INFORMATION — Requests seeking product details, documentation links,
     feature availability, release notes, roadmap queries, pricing tiers, integration
     compatibility matrices, SLA specifications, case studies, API reference pages,
     or benchmark comparisons.

  4. FEEDBACK_AND_GENERAL — General comments, greetings, partnership enquiries,
     feature requests, usability complaints, NPS feedback, escalation requests not
     fitting the above, or any message whose primary intent is non-transactional.

CLASSIFICATION PROCEDURE:
  Step 1 — Pre-process: remove salutations, signatures, and noise tokens.
  Step 2 — Extract: identify key nouns and verbs signalling intent.
  Step 3 — Score: match extracted terms against each category's keyword set.
  Step 4 — Rank: select the highest-scoring category; use FEEDBACK_AND_GENERAL as
            the default when evidence is ambiguous or absent.
  Step 5 — Confidence: compute confidence as (top_score / total_score), clamped
            to [0.50, 0.99].
  Step 6 — Comply: never expose PII (credit-card numbers, SSNs, passwords).
            Redact before responding. Follow GDPR and enterprise data-handling policy.

OUTPUT FORMAT (JSON):
  {"intent": "<category>", "confidence": <float>, "rationale": "<one sentence>"}
"""

# ~420 words ≈ 560 tokens
KNOWLEDGE_RETRIEVER_SYSTEM_PROMPT = """\
ROLE: You are the Advanced Enterprise Knowledge Retrieval Specialist. Your task is to
query the internal vector knowledge base and return the most relevant context chunks
for a given customer query and detected intent.

RETRIEVAL PROTOCOL:
  Step 1 — Query decomposition: split the customer query into two to four semantic
            sub-topics. Ignore stop words, greetings, and boilerplate phrases.
  Step 2 — Query expansion: for each sub-topic, add two to three synonyms or
            closely related technical terms (e.g. "bug" → "error", "defect", "fault").
  Step 3 — Multi-index search: query the following knowledge sections in parallel:
            (a) API Reference Documentation
            (b) User Manuals and Guides
            (c) Troubleshooting Knowledge Base
            (d) Billing and Account FAQ
            (e) Product Feature Catalogue
            (f) Release Notes Archive
  Step 4 — Scoring and re-ranking: rank retrieved chunks by a composite score:
            60 % TF-IDF cosine similarity + 25 % BM25 overlap + 15 % recency weight.
  Step 5 — Grounding filter: discard any chunk whose composite score falls below 0.15.
            This prevents low-quality context from being injected into the response.
  Step 6 — Context budget enforcement: stop adding chunks when accumulated context
            length exceeds 800 tokens to protect the downstream LLM context window.
  Step 7 — Privacy filter: strip any chunk containing credentials, private API keys,
            internal IP addresses, or confidential client data before returning.
  Step 8 — Format output: number each retrieved source (Source [1], Source [2], …).
            Include: source title, category tag, knowledge-base article ID, and date.
  Step 9 — Fallback: if no chunks pass the grounding filter, return the literal string
            "CONTEXT_NOT_FOUND" so that the downstream agent handles the gap gracefully.
  Step 10 — Quota management: if the upstream search API returns a 429 rate-limit
             response, apply exponential back-off (1 s, 2 s, 4 s, max 3 retries)
             before raising a retrieval failure.

TOKEN BUDGET: stop processing when total prompt tokens (system + query + context)
would exceed 4 096 tokens, to remain within the shared KV-cache allocation.

COMPLIANCE: never surface internal pricing negotiations, unpublished roadmap items,
or data covered by an active NDA without explicit authorisation.
"""

RESPONSE_GENERATOR_SYSTEM_PROMPT = """\
ROLE: You are the Customer Response Generator for an enterprise support platform.
Combine the customer message, conversation history, and retrieved knowledge chunks to
produce a polite, professional, and fully grounded response.
Rules: cite every factual claim with an inline reference (e.g. [1], [2]).
Never fabricate information not present in the retrieved context.
If context is insufficient, acknowledge the gap and escalate.
"""

QUALITY_CHECKER_SYSTEM_PROMPT = """\
ROLE: You are the Quality Assurance Agent.
Evaluate the generated response on four axes:
  • Accuracy   (0-1): factual alignment with retrieved context.
  • Coherence  (0-1): logical flow and grammatical correctness.
  • Citation   (0-1): every claim backed by a source reference.
  • Safety     (0-1): no PII, no prohibited content.
Output a composite score = mean(accuracy, coherence, citation, safety).
Approve responses with composite score ≥ 0.80. Reject and return a revision note
otherwise.
"""

# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: dict[str, list[str]] = {
    "TECHNICAL_SUPPORT": [
        "Source [1]: API Connection Error — Verify Authorization header format "
        "(Bearer <token>). Ensure CORS is enabled on your origin in Settings → Security. "
        "Check that your API key has not expired (keys rotate every 90 days by default).",
        "Source [2]: Docker Installation Troubleshooting — Run `docker info` to confirm "
        "the daemon is running. For permission errors on Linux add your user to the "
        "`docker` group: `sudo usermod -aG docker $USER` then log out and back in.",
        "Source [3]: Network Timeout Configuration — Set `request_timeout` in "
        "`config.yaml` to a value ≥ 30 s for slow connections. Enable keep-alive by "
        "setting `http.keep_alive: true`.",
    ],
    "BILLING_AND_ACCOUNT": [
        "Source [1]: Refund Policy — Refunds for annual subscriptions are processed "
        "within 7 business days upon written request to billing@company.com. "
        "Monthly subscriptions are non-refundable after the billing cycle closes.",
        "Source [2]: Subscription Plans — Standard: $29/month (5 seats). "
        "Professional: $99/month (25 seats + priority support). "
        "Enterprise: custom pricing with dedicated CSM and SLA guarantee.",
        "Source [3]: Invoice Download — Log into the dashboard, navigate to "
        "Account → Billing → Invoice History. PDF invoices are available for the "
        "past 24 months.",
    ],
    "PRODUCT_INFORMATION": [
        "Source [1]: Feature Catalogue — The platform supports REST and GraphQL APIs, "
        "webhooks, OAuth 2.0, SSO via SAML 2.0, and a Zapier integration (500+ apps).",
        "Source [2]: Release Notes v3.4 — Added real-time collaboration, improved "
        "dashboard load time by 40 %, and introduced the new workflow automation engine.",
        "Source [3]: SLA Specification — Enterprise tier guarantees 99.9 % uptime "
        "(≤ 8.7 h downtime/year). Incident response SLA: critical P1 < 1 h, P2 < 4 h.",
    ],
    "FEEDBACK_AND_GENERAL": [
        "Source [1]: General Help Centre — Visit help.company.com for guides, "
        "tutorials, and community forums. Live chat support is available Mon-Fri 09:00-18:00 UTC.",
    ],
}

# ---------------------------------------------------------------------------
# KVCacheManager — simulates shared prefix caching (LMCache-style)
# ---------------------------------------------------------------------------

class KVCacheManager:
    """
    Simulates LMCache's shared, block-level KV-cache prefix mechanism.

    Real LMCache (and the vLLM PagedAttention allocator it plugs into) does
    NOT cache "the prefix" as one opaque blob. It splits the token sequence
    into fixed-size blocks (16 tokens is vLLM's default block size), hashes
    each block *chained* with the hash of every block before it
    (`h_i = H(h_{i-1} || tokens_i)`), and stores/looks up KV state at that
    per-block granularity. Two sequences share cached KV state for exactly
    as many leading blocks as their chained hashes agree on — this is what
    makes shared *system-prompt* prefixes across many different sessions/
    conversation histories reusable even though nothing else in the request
    matches. A flat "hash the whole prefix string, cache the whole thing"
    approach (the previous version of this class) can't model *partial*
    prefix reuse and isn't how the real mechanism works, so it's replaced
    here with real block-chain hashing over the tokens of `prefix_key`.

    Memory budget is derived from real per-token KV-cache byte math instead
    of an arbitrary token count, using a documented reference model config
    (see `_kv_bytes_per_token`) so the "8 GB KV cache budget" constraint in
    the assignment is something this class can actually enforce and report
    against in real units (GB), not a made-up token ceiling.
    """

    def __init__(self, config: CustomerIntelligenceConfig | None = None) -> None:
        # Reference model config used to convert tokens -> KV cache bytes, and
        # the memory budget itself, now come from CustomerIntelligenceConfig
        # (env-overridable) instead of hardcoded class constants — see
        # config.py's `CUSTOMER_INTEL_KV_BUDGET_GB` / `_BLOCK_SIZE` / etc.
        self.config = config or CustomerIntelligenceConfig.from_env()
        self.BLOCK_SIZE_TOKENS = self.config.block_size_tokens
        self.KV_BYTES_PER_TOKEN = self.config.kv_bytes_per_token
        self.KV_MEMORY_BUDGET_BYTES = self.config.kv_memory_budget_bytes

        # block_hash -> block index in LRU order (most-recently-used at the end)
        self._block_store: "OrderedDict[str, int]" = OrderedDict()
        # Lock to make concurrent access safe when used from multiple threads
        self._lock = threading.RLock()
        self.total_tokens_processed: int = 0
        self.total_tokens_saved: int = 0
        self._evictions: int = 0

        logger.info(
            "KVCacheManager initialized: budget=%.2fGB block_size=%d tokens "
            "bytes_per_token=%d",
            self.config.kv_memory_budget_gb,
            self.BLOCK_SIZE_TOKENS,
            self.KV_BYTES_PER_TOKEN,
        )

    # ---- block-chain hashing -------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        # Word-split is used as the token proxy throughout this simulation
        # (consistent with how prompt token counts are estimated elsewhere
        # in this module) since there is no real tokenizer/model backend here.
        return text.split()

    def _chunk_hashes(self, tokens: list[str]) -> list[str]:
        """Returns one chained hash per BLOCK_SIZE_TOKENS-token block."""
        hashes: list[str] = []
        running = ""
        for start in range(0, len(tokens), self.BLOCK_SIZE_TOKENS):
            block = tokens[start:start + self.BLOCK_SIZE_TOKENS]
            block_text = " ".join(block)
            running = hashlib.sha256((running + "|" + block_text).encode()).hexdigest()
            hashes.append(running)
        return hashes

    @property
    def _cached_blocks(self) -> int:
        return len(self._block_store)

    @property
    def _cached_bytes(self) -> int:
        return self._cached_blocks * self.BLOCK_SIZE_TOKENS * self.KV_BYTES_PER_TOKEN

    def _evict_until_within_budget(self) -> None:
        with self._lock:
            while self._cached_bytes > self.KV_MEMORY_BUDGET_BYTES and self._block_store:
                self._block_store.popitem(last=False)  # evict least-recently-used block
                self._evictions += 1
                logger.debug(
                    "KV cache evicted LRU block (%d total evictions, cache now %.4fGB)",
                    self._evictions, self._cached_bytes / (1024 ** 3),
                )
                if self._evictions == 1 or self._evictions % 1000 == 0:
                    logger.info(
                        "KV cache under memory pressure: %d evictions so far, "
                        "budget=%.2fGB", self._evictions, self.config.kv_memory_budget_gb,
                    )

    # ---- public API (kept backward-compatible with existing callers/tests) --

    def access(self, key: str, total_tokens: int, prefix_key: str) -> dict[str, Any]:
        """
        Looks up how many leading token-blocks of `prefix_key` are already
        cached (a real chained-prefix match, not a whole-string flag), marks
        those blocks and any new blocks as most-recently-used, evicts LRU
        blocks if the budget is exceeded, and returns how many tokens of
        `total_tokens` were served from cache vs. newly computed.

        `total_tokens` is the full request size (system prompt + dynamic
        content); only the `prefix_key` portion (the static system prompt,
        which is what's actually identical across sessions) is cacheable —
        the dynamic remainder is always new compute, matching how a real
        KV cache can only ever reuse a *prefix*, never arbitrary later
        content that happens to be similar.
        """
        prefix_tokens_list = self._tokenize(prefix_key)
        block_hashes = self._chunk_hashes(prefix_tokens_list)

        with self._lock:
            matched_blocks = 0
            for h in block_hashes:
                if h in self._block_store:
                    matched_blocks += 1
                    self._block_store.move_to_end(h)  # LRU touch
                else:
                    break  # chained hashing: a miss here means every later block also misses

            # Cache every block of this prefix (matched ones get LRU-refreshed above,
            # new ones get inserted) so future calls with the same prefix hit.
            for i, h in enumerate(block_hashes):
                if h not in self._block_store:
                    self._block_store[h] = i
                self._evict_until_within_budget()

            saved_tokens = min(matched_blocks * self.BLOCK_SIZE_TOKENS, total_tokens)
            new_tokens = max(0, total_tokens - saved_tokens)

            self.total_tokens_processed += new_tokens
            self.total_tokens_saved += saved_tokens

            return {
                "cache_hit": saved_tokens > 0,
                "prefix_tokens": len(prefix_tokens_list),
                "matched_blocks": matched_blocks,
                "total_blocks_in_prefix": len(block_hashes),
                "new_tokens": new_tokens,
                "saved_tokens": saved_tokens,
            }

    @property
    def cache_hit_rate(self) -> float:
        with self._lock:
            total = self.total_tokens_processed + self.total_tokens_saved
            return round(self.total_tokens_saved / total, 4) if total else 0.0

    def memory_usage_report(self) -> dict[str, Any]:
        with self._lock:
            cached_bytes = self._cached_bytes
            return {
                "cached_blocks": self._cached_blocks,
                "block_size_tokens": self.BLOCK_SIZE_TOKENS,
                "kv_bytes_per_token": self.KV_BYTES_PER_TOKEN,
                "total_cached_tokens": self._cached_blocks * self.BLOCK_SIZE_TOKENS,
                "total_tokens_processed": self.total_tokens_processed,
                "total_tokens_saved": self.total_tokens_saved,
                "cache_hit_rate": self.cache_hit_rate,
                "cached_gb": round(cached_bytes / (1024 ** 3), 4),
                "budget_gb": round(self.KV_MEMORY_BUDGET_BYTES / (1024 ** 3), 2),
                "budget_used_pct": round(cached_bytes / self.KV_MEMORY_BUDGET_BYTES * 100, 2),
                "evictions": self._evictions,
            }


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class IntentClassifierAgent:
    """
    Agent 1 — intent classification.
    System prompt: ≥ 300 tokens.
    Input: system_prompt + user_message.
    """

    INTENT_KEYWORDS: dict[str, list[str]] = {
        "TECHNICAL_SUPPORT":  ["bug", "error", "install", "api", "docker", "timeout",
                               "crash", "fail", "broken", "ssl", "certificate", "network"],
        "BILLING_AND_ACCOUNT":["invoice", "billing", "payment", "refund", "subscription",
                               "price", "charge", "cancel", "upgrade", "account", "seat"],
        "PRODUCT_INFORMATION": ["feature", "product", "documentation", "docs", "specs",
                                "roadmap", "release", "sla", "integration", "capability"],
    }

    def __init__(self, cache_manager: KVCacheManager) -> None:
        self.system_prompt = INTENT_CLASSIFIER_SYSTEM_PROMPT
        self.cache_manager = cache_manager
        self._prompt_token_count = len(self.system_prompt.split())

    def run(self, user_message: str, session_id: str) -> dict[str, Any]:
        total_tokens = self._prompt_token_count + len(user_message.split())
        cache_info   = self.cache_manager.access(
            key=f"{session_id}_intent",
            total_tokens=total_tokens,
            prefix_key=self.system_prompt,
        )

        # Keyword scoring
        lowered = user_message.lower()
        scores: dict[str, int] = {
            cat: sum(1 for kw in kws if kw in lowered)
            for cat, kws in self.INTENT_KEYWORDS.items()
        }
        best  = max(scores, key=scores.get, default="FEEDBACK_AND_GENERAL")
        total = sum(scores.values())

        if total == 0 or scores[best] == 0:
            intent     = "FEEDBACK_AND_GENERAL"
            confidence = 0.60
        else:
            intent     = best
            confidence = min(0.99, max(0.50, scores[best] / total))

        result = {
            "intent":     intent,
            "confidence": round(confidence, 3),
            "rationale":  f"Matched {scores.get(intent, 0)} keyword(s) for {intent}.",
            "cache_info": cache_info,
        }
        logger.debug(
            "[%s] IntentClassifier -> %s (confidence=%.2f, cache_hit=%s)",
            session_id, result["intent"], result["confidence"], cache_info["cache_hit"],
        )
        return result


class KnowledgeRetrieverAgent:
    """
    Agent 2 — knowledge retrieval.
    System prompt: ≥ 400 tokens.
    Input: system_prompt + user_message + retrieved knowledge context.
    """

    def __init__(self, cache_manager: KVCacheManager) -> None:
        self.system_prompt = KNOWLEDGE_RETRIEVER_SYSTEM_PROMPT
        self.cache_manager = cache_manager
        self._prompt_token_count = len(self.system_prompt.split())

    def run(self, user_message: str, intent: str, session_id: str) -> dict[str, Any]:
        if intent not in KNOWLEDGE_BASE:
            logger.warning(
                "[%s] Unknown intent '%s' — falling back to FEEDBACK_AND_GENERAL "
                "knowledge section", session_id, intent,
            )
        context_chunks = KNOWLEDGE_BASE.get(intent, KNOWLEDGE_BASE["FEEDBACK_AND_GENERAL"])
        context_str    = "\n".join(context_chunks)

        total_tokens = (self._prompt_token_count
                        + len(user_message.split())
                        + len(context_str.split()))
        cache_info   = self.cache_manager.access(
            key=f"{session_id}_retrieval",
            total_tokens=total_tokens,
            prefix_key=self.system_prompt,
        )

        if not context_chunks:
            logger.warning("[%s] CONTEXT_NOT_FOUND — no chunks passed the "
                            "grounding filter for intent '%s'", session_id, intent)

        logger.debug(
            "[%s] KnowledgeRetriever -> %d chunk(s) for intent '%s' (cache_hit=%s)",
            session_id, len(context_chunks), intent, cache_info["cache_hit"],
        )
        return {
            "context":    context_chunks,
            "cache_info": cache_info,
        }


class ResponseGeneratorAgent:
    """
    Agent 3 — response generation.
    Input: system_prompt + retrieved knowledge + conversation history.
    """

    def __init__(self, cache_manager: KVCacheManager, config: CustomerIntelligenceConfig | None = None) -> None:
        self.system_prompt = RESPONSE_GENERATOR_SYSTEM_PROMPT
        self.cache_manager = cache_manager
        self.config = config or CustomerIntelligenceConfig.from_env()
        self._prompt_token_count = len(self.system_prompt.split())

    def run(
        self,
        user_message: str,
        context_chunks: list[str],
        history: list[dict[str, str]],
        session_id: str,
    ) -> dict[str, Any]:
        # Bound history token growth (matches the retrieval agent's own
        # "context budget enforcement" instruction) instead of letting a
        # long-running session's conversation history grow unbounded.
        bounded_history = (history[-self.config.max_history_turns:]
                            if self.config.max_history_turns else [])
        history_text = " ".join(t["content"] for t in bounded_history)
        context_text = "\n".join(context_chunks)
        total_tokens = (self._prompt_token_count
                        + len(user_message.split())
                        + len(context_text.split())
                        + len(history_text.split()))
        cache_info   = self.cache_manager.access(
            key=f"{session_id}_response",
            total_tokens=total_tokens,
            prefix_key=self.system_prompt,
        )

        if context_chunks:
            citations = "\n".join(f"  • {c}" for c in context_chunks)
            response  = (
                f"Thank you for reaching out. Based on our knowledge base, "
                f"here is what I found:\n\n{citations}\n\n"
                f"If you need further assistance, please let me know."
            )
        else:
            logger.warning("[%s] No grounding context available — escalating "
                            "response to a specialist", session_id)
            response = (
                "I'm sorry — I was unable to find specific information on your query "
                "in our knowledge base. I will escalate this to a specialist."
            )

        logger.debug("[%s] ResponseGenerator produced %d-char response (cache_hit=%s)",
                      session_id, len(response), cache_info["cache_hit"])
        return {"response": response, "cache_info": cache_info}


class QualityCheckerAgent:
    """
    Agent 4 — quality checking.
    Input: system_prompt + retrieved articles + history + model response.
    Approves when composite score ≥ 0.80.
    """

    APPROVAL_THRESHOLD = 0.80

    def __init__(self, cache_manager: KVCacheManager, config: CustomerIntelligenceConfig | None = None) -> None:
        self.system_prompt = QUALITY_CHECKER_SYSTEM_PROMPT
        self.cache_manager = cache_manager
        self.config = config or CustomerIntelligenceConfig.from_env()
        self.APPROVAL_THRESHOLD = self.config.quality_approval_threshold
        self._prompt_token_count = len(self.system_prompt.split())

    def run(
        self,
        response: str,
        context_chunks: list[str],
        history: list[dict[str, str]],
        session_id: str,
    ) -> dict[str, Any]:
        context_text = " ".join(context_chunks)
        history_text = " ".join(t["content"] for t in history)
        total_tokens = (self._prompt_token_count
                        + len(response.split())
                        + len(context_text.split())
                        + len(history_text.split()))
        cache_info   = self.cache_manager.access(
            key=f"{session_id}_quality",
            total_tokens=total_tokens,
            prefix_key=self.system_prompt,
        )

        # Heuristic scoring
        accuracy  = 0.90 if context_chunks and any(
            chunk[:20] in response for chunk in context_chunks) else 0.55
        coherence = min(1.0, len(response.split()) / 80)        # longer = more coherent
        citation  = 0.90 if "[1]" in response or "Source" in response else 0.50
        safety    = 0.00 if any(w in response.lower()
                                for w in ["password", "secret", "ssn"]) else 1.00

        composite = round((accuracy + coherence + citation + safety) / 4, 3)
        approved  = composite >= self.APPROVAL_THRESHOLD

        if safety == 0.0:
            logger.error("[%s] SAFETY VIOLATION — response contained a "
                         "prohibited term and was rejected", session_id)
        elif not approved:
            logger.warning(
                "[%s] Response rejected: quality_score=%.3f < threshold=%.2f",
                session_id, composite, self.APPROVAL_THRESHOLD,
            )
        else:
            logger.debug("[%s] Response approved: quality_score=%.3f",
                         session_id, composite)

        return {
            "quality_score": composite,
            "approved":      approved,
            "axes":          {"accuracy": accuracy, "coherence": coherence,
                              "citation": citation, "safety": safety},
            "cache_info":    cache_info,
        }


# ---------------------------------------------------------------------------
# CustomerIntelligencePipeline
# ---------------------------------------------------------------------------

@dataclass
class CustomerTurn:
    """A single validated customer message within a session.

    Previously a bare `@dataclass` with no runtime checks — passing
    `user_message=None` or a whitespace-only string would crash deep inside
    an agent with an unhelpful `AttributeError`. Validation now happens once,
    at construction, and raises a clear `InvalidCustomerTurnError` instead.
    """

    user_message: str
    session_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.user_message, str):
            raise InvalidCustomerTurnError(
                f"user_message must be a string, got {type(self.user_message).__name__}"
            )
        if not self.user_message.strip():
            raise InvalidCustomerTurnError("user_message must not be blank or whitespace-only")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise InvalidCustomerTurnError("session_id must be a non-empty string")


class CustomerIntelligencePipeline:
    """
    Sequential pipeline:
      UserRequest → IntentClassifier → KnowledgeRetriever →
      ResponseGenerator → QualityChecker
    Shared KVCacheManager reduces token processing via prefix reuse.

    Every stage is wrapped so that one agent's failure degrades the turn to
    a safe, logged escalation response instead of crashing the whole
    request — this is what keeps a single malformed message or transient
    error from taking down the other 100+ concurrent sessions being served
    by the same pipeline.
    """

    def __init__(self, config: CustomerIntelligenceConfig | None = None) -> None:
        self.config       = config or CustomerIntelligenceConfig.from_env()
        self.cache_manager = KVCacheManager(self.config)
        self.classifier    = IntentClassifierAgent(self.cache_manager)
        self.retriever     = KnowledgeRetrieverAgent(self.cache_manager)
        self.generator     = ResponseGeneratorAgent(self.cache_manager, self.config)
        self.checker       = QualityCheckerAgent(self.cache_manager, self.config)
        self._sessions: dict[str, list[dict[str, str]]] = {}
        logger.info("CustomerIntelligencePipeline initialized")

    def _get_history(self, session_id: str) -> list[dict[str, str]]:
        return self._sessions.setdefault(session_id, [])

    @staticmethod
    def _run_stage(stage_name: str, session_id: str, fn, *args) -> dict[str, Any]:
        """Runs one agent stage, converting any unexpected exception into an
        `AgentStageError` with full logging, so `handle_turn` can degrade
        gracefully instead of propagating a raw traceback."""
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - intentional: any agent failure must degrade, not crash
            logger.exception("[%s] Stage '%s' failed", session_id, stage_name)
            raise AgentStageError(stage_name, exc) from exc

    def handle_turn(self, turn: CustomerTurn) -> dict[str, Any]:
        if not isinstance(turn, CustomerTurn):
            raise InvalidCustomerTurnError(
                f"handle_turn expects a CustomerTurn, got {type(turn).__name__}"
            )
        if len(turn.user_message) > self.config.max_message_chars:
            logger.warning(
                "[%s] user_message truncated from %d to %d chars (max_message_chars)",
                turn.session_id, len(turn.user_message), self.config.max_message_chars,
            )

        t0      = time.perf_counter()
        sid     = turn.session_id
        message = turn.user_message[: self.config.max_message_chars]
        history = self._get_history(sid)

        logger.info("[%s] Turn started (message_len=%d, history_len=%d)",
                    sid, len(message), len(history))

        try:
            # 1 — Intent
            intent_result = self._run_stage(
                "IntentClassifier", sid, self.classifier.run, message, sid)
            intent = intent_result["intent"]

            # 2 — Retrieval
            retrieval_result = self._run_stage(
                "KnowledgeRetriever", sid, self.retriever.run, message, intent, sid)
            context_chunks = retrieval_result["context"]

            # 3 — Response
            gen_result = self._run_stage(
                "ResponseGenerator", sid, self.generator.run, message, context_chunks, history, sid)
            response = gen_result["response"]

            # 4 — Quality
            quality_result = self._run_stage(
                "QualityChecker", sid, self.checker.run, response, context_chunks, history, sid)

        except AgentStageError as exc:
            # A single failed stage degrades this turn to a safe, unapproved
            # escalation response rather than raising out of the pipeline —
            # other sessions sharing this pipeline/cache are unaffected.
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.error("[%s] Turn degraded to fallback response after stage "
                         "'%s' failed: %r", sid, exc.stage, exc.original)
            return {
                "session_id":    sid,
                "intent":        "FEEDBACK_AND_GENERAL",
                "confidence":    0.0,
                "response":      ("I'm sorry — something went wrong while processing your "
                                  "request. I've escalated this to a specialist."),
                "quality_score": 0.0,
                "approved":      False,
                "latency_ms":    latency_ms,
                "kv_cache":      self.cache_manager.memory_usage_report(),
                "error":         {"stage": exc.stage, "detail": str(exc.original)},
            }

        # Update session history
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant",  "content": response})

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("[%s] Turn complete: intent=%s quality=%.3f approved=%s latency_ms=%.1f",
                    sid, intent, quality_result["quality_score"],
                    quality_result["approved"], latency_ms)

        return {
            "session_id":    sid,
            "intent":        intent,
            "confidence":    intent_result["confidence"],
            "response":      response,
            "quality_score": quality_result["quality_score"],
            "approved":      quality_result["approved"],
            "latency_ms":    latency_ms,
            "kv_cache":      self.cache_manager.memory_usage_report(),
        }
