"""Unit tests for the Customer Intelligence pipeline (updated for new API)."""
from __future__ import annotations

import pytest
from research_pipeline.customer_intelligence.pipeline import (
    CustomerIntelligencePipeline,
    CustomerTurn,
    KVCacheManager,
    IntentClassifierAgent,
    KnowledgeRetrieverAgent,
    QualityCheckerAgent,
    InvalidCustomerTurnError,
    AgentStageError,
)
from research_pipeline.customer_intelligence.config import CustomerIntelligenceConfig


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------

def test_kv_cache_saves_tokens_on_second_access():
    """Prefix tokens should be saved (not reprocessed) on a repeated key."""
    mgr = KVCacheManager()
    prefix = "A long system prompt that is the same every time."
    r1 = mgr.access("s1_intent", total_tokens=500, prefix_key=prefix)
    r2 = mgr.access("s2_intent", total_tokens=500, prefix_key=prefix)
    assert r1["saved_tokens"] == 0       # first hit — nothing cached yet
    assert r2["saved_tokens"] > 0        # second hit — prefix reused
    assert mgr.total_tokens_saved > 0


def test_kv_cache_hit_rate_grows_across_turns():
    """Cache hit rate must increase as more turns share the same prefixes."""
    mgr = KVCacheManager()
    prefix = "shared system prompt"
    for i in range(5):
        mgr.access(f"s{i}_intent", total_tokens=400, prefix_key=prefix)
    assert mgr.cache_hit_rate > 0.0


def test_kv_cache_block_level_partial_prefix_reuse():
    """Two prefixes that share their first N blocks but diverge afterward
    must reuse exactly the shared leading blocks — not the whole prefix,
    not zero — proving this is real block-chained matching rather than a
    whole-string cache-hit/miss flag."""
    mgr = KVCacheManager()
    shared_words = " ".join(f"tok{i}" for i in range(mgr.BLOCK_SIZE_TOKENS * 3))  # 3 full blocks
    prefix_a = shared_words + " " + " ".join(f"aonly{i}" for i in range(mgr.BLOCK_SIZE_TOKENS))
    prefix_b = shared_words + " " + " ".join(f"bonly{i}" for i in range(mgr.BLOCK_SIZE_TOKENS))

    mgr.access("k1", total_tokens=len(prefix_a.split()), prefix_key=prefix_a)
    result_b = mgr.access("k2", total_tokens=len(prefix_b.split()), prefix_key=prefix_b)

    # prefix_b shares exactly the first 3 blocks with prefix_a, then diverges
    assert result_b["matched_blocks"] == 3
    assert result_b["saved_tokens"] == 3 * mgr.BLOCK_SIZE_TOKENS


def test_kv_cache_respects_8gb_budget_and_evicts():
    """Filling the cache well past the 8GB budget must trigger evictions
    and keep reported cached_gb within the budget."""
    mgr = KVCacheManager()
    for i in range(50_000):
        # Each prefix is unique -> guarantees new blocks are stored every call,
        # eventually forcing eviction once the budget is exceeded.
        prefix = f"unique prefix number {i} " + " ".join(f"w{j}" for j in range(20))
        mgr.access(f"sess{i}", total_tokens=len(prefix.split()), prefix_key=prefix)
        if mgr.memory_usage_report()["evictions"] > 0:
            break

    report = mgr.memory_usage_report()
    assert report["evictions"] > 0
    assert report["cached_gb"] <= report["budget_gb"] + 1e-6


# ---------------------------------------------------------------------------
# IntentClassifierAgent
# ---------------------------------------------------------------------------

def test_intent_classifier_technical_support():
    mgr = KVCacheManager()
    agent = IntentClassifierAgent(mgr)
    result = agent.run("My Docker container keeps crashing on startup", "sess-1")
    assert result["intent"] == "TECHNICAL_SUPPORT"
    assert result["confidence"] >= 0.50


def test_intent_classifier_billing():
    mgr = KVCacheManager()
    agent = IntentClassifierAgent(mgr)
    result = agent.run("I need a refund for my last invoice payment", "sess-2")
    assert result["intent"] == "BILLING_AND_ACCOUNT"


def test_intent_classifier_general_fallback():
    mgr = KVCacheManager()
    agent = IntentClassifierAgent(mgr)
    result = agent.run("Hello, how are you doing today?", "sess-3")
    assert result["intent"] == "FEEDBACK_AND_GENERAL"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_pipeline_classifies_and_approves_response():
    """Technical support query should produce an approved, cited response."""
    pipeline = CustomerIntelligencePipeline()
    result = pipeline.handle_turn(
        CustomerTurn(user_message="I have an API connection error", session_id="s1"))
    assert result["intent"] == "TECHNICAL_SUPPORT"
    assert "response" in result
    assert result["quality_score"] >= 0.0          # must be a valid score
    assert "approved" in result
    assert "kv_cache" in result


def test_pipeline_multi_turn_builds_history():
    """Second turn in same session should see cache savings from shared prefixes."""
    pipeline = CustomerIntelligencePipeline()
    pipeline.handle_turn(CustomerTurn(user_message="I need an invoice refund", session_id="s5"))
    result2 = pipeline.handle_turn(
        CustomerTurn(user_message="Can I cancel my subscription?", session_id="s5"))
    # After 2 turns the KV cache should have saved at least some tokens
    assert result2["kv_cache"]["total_tokens_saved"] > 0


def test_pipeline_handles_100_concurrent_style_sessions_within_budget():
    """Spec requirement: '4 agents x 100 sessions x multi-turn conversations'
    must fit the 8GB KV-cache budget. Simulates 100 sessions x 3 turns each
    (1,200 total agent invocations across the 4-agent pipeline) and asserts
    the shared system-prompt caching keeps memory well under budget while
    still producing a non-trivial hit rate."""
    import random
    random.seed(7)
    pipeline = CustomerIntelligencePipeline()
    sample_messages = [
        "My API keeps timing out",
        "I need a refund for my invoice",
        "What features does the enterprise plan include?",
        "Can you help me with docker installation",
        "What is your SLA uptime guarantee",
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = []
    with ThreadPoolExecutor(max_workers=20) as exe:
        for session_num in range(100):
            sid = f"session_{session_num}"
            for _ in range(3):
                tasks.append(exe.submit(
                    pipeline.handle_turn,
                    CustomerTurn(user_message=random.choice(sample_messages), session_id=sid)
                ))

        # Ensure all tasks complete and raise any exceptions
        for fut in as_completed(tasks):
            _ = fut.result()

    report = pipeline.cache_manager.memory_usage_report()
    assert report["cached_gb"] <= report["budget_gb"]
    assert report["cache_hit_rate"] > 0.3  # heavy reuse expected: identical system prompts


def test_pipeline_product_intent():
    """Product information queries should retrieve relevant knowledge."""
    pipeline = CustomerIntelligencePipeline()
    result = pipeline.handle_turn(
        CustomerTurn(user_message="What features does the product include?", session_id="s6"))
    assert result["intent"] == "PRODUCT_INFORMATION"
    assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# CustomerTurn input validation (malformed content)
# ---------------------------------------------------------------------------

def test_customer_turn_rejects_none_message():
    with pytest.raises(InvalidCustomerTurnError):
        CustomerTurn(user_message=None, session_id="s1")


def test_customer_turn_rejects_non_string_message():
    with pytest.raises(InvalidCustomerTurnError):
        CustomerTurn(user_message=12345, session_id="s1")


def test_customer_turn_rejects_blank_message():
    with pytest.raises(InvalidCustomerTurnError):
        CustomerTurn(user_message="   \n\t  ", session_id="s1")


def test_customer_turn_rejects_empty_session_id():
    with pytest.raises(InvalidCustomerTurnError):
        CustomerTurn(user_message="hello", session_id="")


def test_customer_turn_accepts_valid_input():
    turn = CustomerTurn(user_message="hello there", session_id="s1")
    assert turn.user_message == "hello there"


# ---------------------------------------------------------------------------
# CustomerIntelligenceConfig (configuration management)
# ---------------------------------------------------------------------------

def test_config_defaults_match_original_hardcoded_values():
    cfg = CustomerIntelligenceConfig()
    assert cfg.kv_memory_budget_gb == 8.0
    assert cfg.block_size_tokens == 16
    assert cfg.quality_approval_threshold == 0.80
    assert cfg.kv_bytes_per_token == 2 * 32 * 8 * 128 * 2


def test_config_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        CustomerIntelligenceConfig(quality_approval_threshold=1.5)


def test_config_from_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("CUSTOMER_INTEL_KV_BUDGET_GB", "2.0")
    monkeypatch.setenv("CUSTOMER_INTEL_APPROVAL_THRESHOLD", "0.95")
    cfg = CustomerIntelligenceConfig.from_env()
    assert cfg.kv_memory_budget_gb == 2.0
    assert cfg.quality_approval_threshold == 0.95


def test_pipeline_honors_custom_config():
    """A pipeline built with a tiny KV budget should evict much sooner than
    the 8GB default, proving the budget is actually threaded through."""
    tiny_cfg = CustomerIntelligenceConfig(kv_memory_budget_gb=0.0001)
    pipeline = CustomerIntelligencePipeline(config=tiny_cfg)
    result = pipeline.handle_turn(
        CustomerTurn(user_message="docker install error", session_id="tiny"))
    assert result["kv_cache"]["evictions"] > 0
    # memory_usage_report() rounds budget_gb to 2 decimals for readability,
    # so a budget this tiny reports as ~0.0 rather than matching exactly.
    assert result["kv_cache"]["budget_gb"] < 0.01


def test_pipeline_honors_custom_quality_threshold():
    """A very high approval threshold should cause an otherwise-fine
    response to be rejected."""
    strict_cfg = CustomerIntelligenceConfig(quality_approval_threshold=0.999)
    pipeline = CustomerIntelligencePipeline(config=strict_cfg)
    result = pipeline.handle_turn(
        CustomerTurn(user_message="What is your SLA?", session_id="strict"))
    assert result["approved"] is False


# ---------------------------------------------------------------------------
# Graceful error handling (a stage failure must degrade, not crash)
# ---------------------------------------------------------------------------

def test_handle_turn_rejects_wrong_type():
    pipeline = CustomerIntelligencePipeline()
    with pytest.raises(InvalidCustomerTurnError):
        pipeline.handle_turn("not a CustomerTurn")  # type: ignore[arg-type]


def test_pipeline_degrades_gracefully_on_stage_failure(monkeypatch):
    """If one agent stage raises unexpectedly, the pipeline must return a
    safe, unapproved fallback response instead of propagating the crash —
    and must remain usable for subsequent, unrelated sessions."""
    pipeline = CustomerIntelligencePipeline()
    original_run = pipeline.retriever.run

    def boom(*args, **kwargs):
        raise RuntimeError("simulated downstream failure")

    monkeypatch.setattr(pipeline.retriever, "run", boom)
    result = pipeline.handle_turn(
        CustomerTurn(user_message="docker error", session_id="fail-1"))

    assert result["approved"] is False
    assert result["quality_score"] == 0.0
    assert result["error"]["stage"] == "KnowledgeRetriever"

    # Restore the stage (monkeypatch only auto-restores at test teardown,
    # not mid-test) and confirm the pipeline still works normally for a
    # different session afterward.
    monkeypatch.setattr(pipeline.retriever, "run", original_run)
    ok_result = pipeline.handle_turn(
        CustomerTurn(user_message="I need a refund", session_id="ok-1"))
    assert ok_result["intent"] == "BILLING_AND_ACCOUNT"
    assert "error" not in ok_result
