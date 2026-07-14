"""Tests for the real LMCache+vLLM production integration path.

This sandbox has no GPU, so `lmcache`/`vllm` are expected to be either
absent or non-functional here — these tests assert the module degrades
gracefully (clear errors, no crashes, correct availability reporting)
rather than testing the GPU-only code paths themselves, which can only be
exercised on real hardware.
"""
from __future__ import annotations

import pytest

from research_pipeline.customer_intelligence import lmcache_production as lp


def test_is_real_lmcache_available_false_without_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert lp.is_real_lmcache_available() is False


def test_build_lmcache_config_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(lp, "_LMCACHE_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="lmcache"):
        lp.build_lmcache_config()


def test_build_vllm_engine_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(lp, "_VLLM_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="vllm"):
        lp.build_vllm_engine()
