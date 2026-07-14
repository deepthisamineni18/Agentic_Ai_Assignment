"""Configuration utilities for the Active Learning pipeline.

Reads values from CLI args (when provided) and environment variables
(`AL_TOKEN_BUDGET`, `AL_TARGET_ACCURACY`, `AL_BATCH_SIZE`, `AL_NUM_SAMPLES`,
`AL_ANNOTATION_CONFIDENCE_TARGET`) with sensible defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class ActiveLearningConfig:
    token_budget: int = 50_000
    target_accuracy: float = 0.85
    batch_size: int = 20
    num_samples: int = 300
    annotation_confidence_target: float = 0.8


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def load_from_args(args: Any) -> ActiveLearningConfig:
    """Load configuration from an argparse `args` object and environment.

    CLI args override environment variables when present. If neither is
    provided, defaults in `ActiveLearningConfig` are used.
    """
    cfg = ActiveLearningConfig()

    # token budget
    if getattr(args, "al_token_budget", None) is not None:
        cfg.token_budget = int(args.al_token_budget)
    else:
        cfg.token_budget = _env_int("AL_TOKEN_BUDGET", cfg.token_budget)

    # target accuracy
    if getattr(args, "al_target_accuracy", None) is not None:
        cfg.target_accuracy = float(args.al_target_accuracy)
    else:
        cfg.target_accuracy = _env_float("AL_TARGET_ACCURACY", cfg.target_accuracy)

    # batch size
    if getattr(args, "al_batch_size", None) is not None:
        cfg.batch_size = int(args.al_batch_size)
    else:
        cfg.batch_size = _env_int("AL_BATCH_SIZE", cfg.batch_size)

    # num samples
    if getattr(args, "al_num_samples", None) is not None:
        cfg.num_samples = int(args.al_num_samples)
    else:
        cfg.num_samples = _env_int("AL_NUM_SAMPLES", cfg.num_samples)

    # annotation confidence target
    if getattr(args, "al_annotation_confidence_target", None) is not None:
        cfg.annotation_confidence_target = float(args.al_annotation_confidence_target)
    else:
        cfg.annotation_confidence_target = _env_float(
            "AL_ANNOTATION_CONFIDENCE_TARGET", cfg.annotation_confidence_target
        )

    return cfg
