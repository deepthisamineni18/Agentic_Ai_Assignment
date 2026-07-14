"""
Configuration for the Customer Intelligence pipeline.

Everything that was previously a hardcoded class constant in `pipeline.py`
(the 8GB KV budget, the 16-token block size, the reference model's KV-byte
formula, the quality-approval threshold, history/message size limits) now
lives here and can be overridden via environment variables without touching
source code — e.g. to point the KV-byte formula at a different model shape,
or to loosen the approval threshold for a staging environment.

Every env var is optional; defaults reproduce the exact behavior the
pipeline shipped with originally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_ENV_PREFIX = "CUSTOMER_INTEL_"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class CustomerIntelligenceConfig:
    """Immutable runtime configuration for the pipeline and its KV cache.

    Env vars (all optional, prefixed `CUSTOMER_INTEL_`):
        KV_BUDGET_GB        — total KV-cache memory budget in GB   (default 8.0)
        BLOCK_SIZE          — vLLM-style prefix block size, tokens (default 16)
        NUM_LAYERS          — reference model transformer layers   (default 32)
        NUM_KV_HEADS        — reference model KV heads (GQA)       (default 8)
        HEAD_DIM            — reference model attention head dim  (default 128)
        DTYPE_BYTES         — KV dtype size in bytes (2 = fp16)    (default 2)
        APPROVAL_THRESHOLD  — quality-checker approval cutoff     (default 0.80)
        MAX_MESSAGE_CHARS   — max accepted length of a user msg   (default 4000)
        MAX_HISTORY_TURNS   — turns of history fed to downstream
                               agents, bounding token growth       (default 20)
    """

    kv_memory_budget_gb: float = 8.0
    block_size_tokens: int = 16
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2
    quality_approval_threshold: float = 0.80
    max_message_chars: int = 4000
    max_history_turns: int = 20

    def __post_init__(self) -> None:
        if self.kv_memory_budget_gb <= 0:
            raise ValueError("kv_memory_budget_gb must be positive")
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if not 0.0 <= self.quality_approval_threshold <= 1.0:
            raise ValueError("quality_approval_threshold must be within [0, 1]")
        if self.max_message_chars <= 0:
            raise ValueError("max_message_chars must be positive")
        if self.max_history_turns < 0:
            raise ValueError("max_history_turns must be >= 0")

    @property
    def kv_memory_budget_bytes(self) -> int:
        return int(self.kv_memory_budget_gb * (1024 ** 3))

    @property
    def kv_bytes_per_token(self) -> int:
        # bytes_per_token = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @classmethod
    def from_env(cls) -> "CustomerIntelligenceConfig":
        return cls(
            kv_memory_budget_gb=_float_env("KV_BUDGET_GB", 8.0),
            block_size_tokens=_int_env("BLOCK_SIZE", 16),
            num_layers=_int_env("NUM_LAYERS", 32),
            num_kv_heads=_int_env("NUM_KV_HEADS", 8),
            head_dim=_int_env("HEAD_DIM", 128),
            dtype_bytes=_int_env("DTYPE_BYTES", 2),
            quality_approval_threshold=_float_env("APPROVAL_THRESHOLD", 0.80),
            max_message_chars=_int_env("MAX_MESSAGE_CHARS", 4000),
            max_history_turns=_int_env("MAX_HISTORY_TURNS", 20),
        )
