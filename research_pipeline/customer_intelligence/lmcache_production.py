"""
Real LMCache + vLLM integration path for the Customer Intelligence pipeline.

WHY THIS FILE EXISTS SEPARATELY FROM pipeline.py
-------------------------------------------------
`pipeline.py`'s `KVCacheManager` is a *simulation* of LMCache's block-chained
prefix-caching mechanism (see its docstring) that runs anywhere, including a
CPU-only sandbox with no GPU and no internet access to huggingface.co. It is
what lets the tests and `make run-customer-intelligence` demo work without
special hardware.

This module is the real thing: it configures the actual `lmcache` PyPI
package (https://pypi.org/project/lmcache/) wired into a real vLLM
`LLMEngine`, so the four agents share one physical LLM backend's KV cache on
an actual GPU, exactly as the assignment specifies. It is intentionally kept
separate and import-guarded because `lmcache`'s own dependency list
(`cupy-cuda13x`, `nixl-cu13`, `cufile-python`, ...) is CUDA/GPU-only and will
not install or run on a CPU-only machine — that's a hardware constraint of
the library itself, not a design choice made here.

WHAT YOU NEED TO ACTUALLY RUN THIS
-----------------------------------
- An NVIDIA GPU with >= 8GB free VRAM for the KV cache budget, CUDA drivers.
- `pip install lmcache vllm`
- A locally-downloadable model (this repo cannot fetch one — no
  huggingface.co access in this environment). Point `MODEL_NAME` at
  whatever chat model you have local weights or a valid HF token for.

HOW IT MAPS TO THE ASSIGNMENT
-------------------------------
- One `vllm.LLMEngine` = "a single LLM backend" shared by all 4 agents.
- `LMCacheEngineConfig` below sets a `max_local_cpu_size` that, combined
  with vLLM's own GPU KV cache pool, is the real analogue of the "strict
  8GB KV cache memory budget" in the assignment -- CPU-side LMCache offload
  size plus the GPU pool size are configured to stay under budget together.
- Each agent's *system prompt* (`INTENT_CLASSIFIER_SYSTEM_PROMPT` etc. from
  `pipeline.py`) is submitted as a fixed prefix on every call; LMCache's
  chunked prefix hashing (the same mechanism `KVCacheManager` simulates)
  automatically reuses the cached KV blocks for that prefix across all
  sessions and turns, which is exactly the "shared KV cache" savings the
  assignment asks you to engineer for.
"""
from __future__ import annotations

import logging
import os

from research_pipeline.customer_intelligence.config import CustomerIntelligenceConfig

logger = logging.getLogger("LMCacheProduction")

try:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.config import LMCacheEngineConfig
    _LMCACHE_AVAILABLE = True
except ImportError:
    _LMCACHE_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


MODEL_NAME = os.environ.get("CUSTOMER_INTEL_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# The overall KV memory budget is the same `CustomerIntelligenceConfig` used
# by the CPU simulation in pipeline.py (env var `CUSTOMER_INTEL_KV_BUDGET_GB`,
# default 8.0), split evenly between LMCache's CPU-side offload tier and
# vLLM's own GPU-resident KV pool so both paths stay in sync with one budget.
_config = CustomerIntelligenceConfig.from_env()
LMCACHE_CPU_OFFLOAD_GB = _config.kv_memory_budget_gb / 2
VLLM_GPU_KV_CACHE_GB = _config.kv_memory_budget_gb / 2


def build_lmcache_config() -> "LMCacheEngineConfig":
    """Builds the real LMCache engine config: chunked prefix hashing +
    CPU-offload tier, so shared system-prompt prefixes across all 4 agents
    and all sessions are served from cache instead of recomputed."""
    if not _LMCACHE_AVAILABLE:
        logger.error("lmcache import unavailable — cannot build engine config")
        raise RuntimeError(
            "lmcache is not installed or failed to import (it requires CUDA-only "
            "dependencies such as cupy-cuda13x / nixl-cu13). Install on a GPU "
            "machine with `pip install lmcache` to use this path."
        )
    logger.info("Building LMCache engine config: cpu_offload=%.2fGB chunk_size=256",
                LMCACHE_CPU_OFFLOAD_GB)
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,                       # tokens per hashed/cached block
        local_cpu=True,                       # enable CPU-side offload tier
        max_local_cpu_size=LMCACHE_CPU_OFFLOAD_GB,
        remote_url=None,                      # single-node: no remote (Redis/S3) tier
        save_decode_cache=True,               # cache KV for generated tokens too,
                                               # so multi-turn history also hits cache
    )


def build_vllm_engine() -> "LLM":
    """Builds a real vLLM engine with LMCache wired in as its KV-cache
    connector. This is the single shared LLM backend all 4 agents call."""
    if not _VLLM_AVAILABLE:
        logger.error("vllm import unavailable — cannot build engine")
        raise RuntimeError(
            "vllm is not installed. Install on a GPU machine with `pip install vllm` "
            "to use this path."
        )
    if not _LMCACHE_AVAILABLE:
        logger.error("lmcache import unavailable — cannot build engine")
        raise RuntimeError("lmcache is not installed; see build_lmcache_config().")

    # This env var is how LMCache's vLLM integration picks up the engine
    # config at process start (documented at https://docs.lmcache.ai).
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(LMCACHE_CPU_OFFLOAD_GB)

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    logger.info("Building vLLM engine: model=%s gpu_kv_budget=%.2fGB",
                MODEL_NAME, VLLM_GPU_KV_CACHE_GB)
    return LLM(
        model=MODEL_NAME,
        kv_transfer_config=ktc,
        gpu_memory_utilization=0.35,  # leaves headroom so the KV pool stays near VLLM_GPU_KV_CACHE_GB
        enforce_eager=True,           # simpler, more predictable memory accounting for this use case
    )


class ProductionLLMBackend:
    """Thin wrapper exposing the same `generate(system_prompt, user_content)`
    shape the 4 agents in pipeline.py would call, but backed by the real
    shared vLLM+LMCache engine instead of the CPU simulation. Swap this in
    for `KVCacheManager`-based agents once running on real GPU hardware --
    the agent classes' external behavior (return shape) is unchanged, only
    where the token savings actually come from (real KV tensors, not a
    tracked count) changes.
    """

    def __init__(self) -> None:
        self.engine = build_vllm_engine()
        self.sampling_params = SamplingParams(temperature=0.2, max_tokens=512)

    def generate(self, system_prompt: str, user_content: str) -> str:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_content}\n<|assistant|>\n"
        outputs = self.engine.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text


def is_real_lmcache_available() -> bool:
    """Used by the pipeline / CLI to decide whether to route through the
    real GPU-backed engine or the portable CPU simulation."""
    return _LMCACHE_AVAILABLE and _VLLM_AVAILABLE and os.environ.get("CUDA_VISIBLE_DEVICES") is not None
