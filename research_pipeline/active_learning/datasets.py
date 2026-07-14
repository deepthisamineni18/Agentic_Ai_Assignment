"""Dataset utilities for Active Learning pipeline.

Provides an optional real dataset loader (AG News via `datasets` or `torchtext`) and
falls back to the repo's synthetic generator when real datasets aren't available.
"""
from __future__ import annotations

from typing import Any, List
import logging

logger = logging.getLogger("al.datasets")


def load_ag_news(num_samples: int = 500) -> List[dict[str, Any]]:
    """Attempt to load AG News and return the expected record format.

    Tries HuggingFace `datasets` first, then `torchtext`. If neither is
    available, raises ImportError.
    """
    # HF `datasets` >= 2.20 removed support for community loading *scripts*.
    # The old repo id "ag_news" is script-based and now fails (404 on
    # ag_news.py / requires trust_remote_code) even though the SDK and
    # network access both work fine. "fancyzhx/ag_news" is the canonical
    # parquet-backed mirror of the same data and loads without a script, so
    # it's tried first. "ag_news" is kept as a second attempt for older
    # `datasets` versions that still support script-based loading.
    for repo_id in ("fancyzhx/ag_news", "ag_news"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo_id, split="train")
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed=42)
            records = []
            for i, item in enumerate(ds.select(range(min(len(ds), num_samples)))):
                label = int(item["label"])
                text = (item.get("text") or "").strip()
                # AG News labels: 0=World,1=Sports,2=Business,3=Sci/Tech.
                # The pipeline's own taxonomy (see CLASSES/KEYWORDS in
                # annotation_pipeline.py) is technology/sports/politics/
                # entertainment/science — a different 5-way scheme, not
                # AG News's 4-way one. Mapping straight to "world"/"business"
                # would leave true_label holding categories the Annotator's
                # prompt and keyword fallback can never actually produce,
                # which makes true_label meaningless. Map onto the closest
                # existing class instead so ground truth stays comparable to
                # what gets predicted: World news skews toward government/
                # elections -> politics; Business skews toward companies/
                # markets/tech industry -> technology.
                label_map = {0: "politics", 1: "sports", 2: "technology", 3: "science"}
                records.append({
                    "id": f"ag_{i}",
                    "text": text,
                    "true_label": label_map.get(label, "world"),
                    "label": None,
                    "confidence": None,
                })
            logger.info("Loaded %d AG News samples from '%s'.", len(records), repo_id)
            return records
        except Exception as e:  # datasets not installed or download failed
            logger.debug("datasets.load_dataset(%r) unavailable or failed: %s", repo_id, e)

    try:
        # torchtext fallback
        from torchtext.datasets import AG_NEWS
        it = AG_NEWS(split="train")
        records = []
        for i, (label, text) in enumerate(it):
            if i >= num_samples:
                break
            # torchtext's AG_NEWS labels are 1-indexed (1=World..4=Sci/Tech);
            # same remap rationale as the `datasets` path above.
            label_map = {1: "politics", 2: "sports", 3: "technology", 4: "science"}
            records.append({
                "id": f"ag_{i}",
                "text": text if isinstance(text, str) else " ".join(text),
                "true_label": label_map.get(int(label), "world"),
                "label": None,
                "confidence": None,
            })
        return records
    except Exception as e:
        logger.debug("torchtext AG_NEWS unavailable or failed: %s", e)

    raise ImportError("No AG News loader available (install `datasets` or `torchtext`)" )


def get_active_learning_dataset(use_real: bool = False, num_samples: int = 500, synthetic_loader=None):
    """Return a dataset compatible with the Active Learning pipeline.

    If `use_real` is True, attempts to load AG News; if unavailable, falls
    back to the provided `synthetic_loader` (callable) or raises.
    """
    if use_real:
        try:
            return load_ag_news(num_samples=num_samples)
        except Exception as e:
            logger.warning("AG News unavailable; falling back to synthetic dataset: %s", e)
    # fallback: call the synthetic loader if provided, else raise
    if synthetic_loader is not None:
        return synthetic_loader(num_samples=num_samples)
    raise RuntimeError("No dataset available for Active Learning")
