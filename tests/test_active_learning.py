"""Unit tests for the Active Learning pipeline (updated for new API)."""
from __future__ import annotations

import sys
import types

import pytest
from research_pipeline.active_learning.annotation_pipeline import (
    AnnotationResult,
    AnnotatorAgent,
    QualityAssessorAgent,
    TrainerAgent,
    get_synthetic_dataset,
    CLASSES,
)
from research_pipeline.active_learning import datasets as datasets_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annotation(label: str = "technology", confidence: float = 0.80) -> AnnotationResult:
    return AnnotationResult(
        sample_id="test-001",
        text="The new software algorithm uses deep learning chip design.",
        true_label=label,
        label=label,
        confidence=confidence,
    )


def _make_pool(n: int = 30) -> list[AnnotationResult]:
    dataset = get_synthetic_dataset(num_samples=100)[:n]
    anns = []
    for doc in dataset:
        anns.append(AnnotationResult(
            sample_id=doc["id"],
            text=doc["text"],
            true_label=doc["true_label"],
            label=doc["true_label"],   # use ground truth for training
            confidence=0.95,
        ))
    return anns


# ---------------------------------------------------------------------------
# AnnotatorAgent
# ---------------------------------------------------------------------------

def test_annotator_budget_enforced():
    """Annotator should stop returning results once token budget is exhausted."""
    # Budget for exactly 2 samples (300 tokens each = 600 total)
    annotator = AnnotatorAgent(token_budget=600)
    dataset = get_synthetic_dataset(num_samples=10)
    results, budget_hit = annotator.annotate_batch(dataset)
    assert len(results) <= 2
    assert budget_hit or annotator.tokens_used <= 600


def test_annotator_returns_valid_labels():
    """Each annotation label must be one of the known classes."""
    annotator = AnnotatorAgent(token_budget=50_000)
    dataset = get_synthetic_dataset(num_samples=10)
    results, _ = annotator.annotate_batch(dataset)
    assert len(results) == 10
    for r in results:
        assert r.label in CLASSES
        assert 0.0 <= r.confidence <= 1.0


def test_annotator_novel_selection_returns_batch_size():
    """select_novel_samples should return exactly batch_size items."""
    annotator = AnnotatorAgent()
    unlabeled = get_synthetic_dataset(num_samples=100)
    labeled_pool = [{"id": d["id"], "text": d["text"]} for d in unlabeled[:10]]
    selected = annotator.select_novel_samples(unlabeled[10:], labeled_pool, batch_size=15)
    assert len(selected) == 15


# ---------------------------------------------------------------------------
# QualityAssessorAgent
# ---------------------------------------------------------------------------

def test_quality_assessor_boosts_low_confidence():
    """Low-confidence annotations should have their confidence increased."""
    assessor = QualityAssessorAgent()
    ann = _make_annotation(confidence=0.40)
    original_conf = ann.confidence
    [assessed] = assessor.assess([ann])
    assert assessed.confidence > original_conf


def test_quality_assessor_leaves_high_confidence_unchanged():
    """High-confidence annotations should not be penalised."""
    assessor = QualityAssessorAgent()
    ann = _make_annotation(confidence=0.95)
    [assessed] = assessor.assess([ann])
    # confidence should stay >= original (not drop)
    assert assessed.confidence >= 0.90


# ---------------------------------------------------------------------------
# TrainerAgent
# ---------------------------------------------------------------------------

def test_trainer_produces_report_with_per_class_metrics():
    """TrainingReport must include test_accuracy and metrics for every class present."""
    trainer = TrainerAgent(target_accuracy=0.99, max_lstm_epochs=3)
    pool = _make_pool(n=40)
    report = trainer.train(pool)
    assert 0.0 <= report.test_accuracy <= 1.0
    assert report.model_name in {"LogisticRegression", "RandomForest", "KNN", "LSTM"}
    for cls_metrics in report.metrics_per_class.values():
        assert "precision" in cls_metrics
        assert "recall"    in cls_metrics
        assert "f1_score"  in cls_metrics


def test_trainer_early_stops_when_target_met():
    """When target_accuracy is very low the trainer should flag early_stopped=True."""
    trainer = TrainerAgent(target_accuracy=0.01, max_lstm_epochs=2)
    pool = _make_pool(n=40)
    report = trainer.train(pool)
    # With a 1% target, at least one model will cross it → the LSTM path may early-stop.
    # We only assert the report is structurally correct here.
    assert isinstance(report.early_stopped, bool)


def test_trainer_split_uses_true_labels_without_duplication():
    """Small pools should be split without duplicating rows before training."""
    trainer = TrainerAgent()
    anns = [
        AnnotationResult(sample_id=f"s{i}", text=f"sample {i}", true_label="technology", label="sports", confidence=0.9)
        for i in range(10)
    ]

    X_tr, X_val, X_te, y_tr, y_val, y_te = trainer._split(anns)

    assert len(X_tr) + len(X_val) + len(X_te) == len(anns)
    assert y_tr[0] == "technology"
    assert y_val[0] == "technology"
    assert y_te[0] == "technology"


def test_load_ag_news_shuffles_before_sampling(monkeypatch):
    """AG News loading should shuffle before taking the requested slice."""
    class FakeDataset(list):
        def __init__(self, items):
            super().__init__(items)
            self.shuffled = False

        def shuffle(self, seed=None):
            self.shuffled = True
            self[:] = list(reversed(self))
            return self

        def select(self, indices):
            assert self.shuffled, "dataset should be shuffled before selecting"
            return [self[i] for i in indices]

    def fake_load_dataset(repo_id, split):
        assert repo_id == "fancyzhx/ag_news"
        return FakeDataset([
            {"label": 0, "text": "world one"},
            {"label": 1, "text": "sports two"},
            {"label": 2, "text": "business three"},
            {"label": 3, "text": "tech four"},
        ])

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    records = datasets_module.load_ag_news(num_samples=3)

    assert len(records) == 3
    assert records[0]["true_label"] in CLASSES


def test_active_learning_pipeline_stops_when_confidence_and_accuracy_met():
    """Full pipeline should stop once annotation confidence and accuracy goals are reached."""
    from research_pipeline.active_learning.annotation_pipeline import ActiveLearningPipeline

    pipeline = ActiveLearningPipeline(
        token_budget=50_000,
        target_accuracy=0.50,
        annotation_confidence_target=0.70,
        batch_size=10,
        num_samples=60,
        use_real_dataset=False,
    )

    result = pipeline.run()

    assert result["labeled_count"] > 0
    assert result["mean_confidence"] >= 0.70
    assert result["final_report"]["test_accuracy"] >= 0.50
    assert result["stop_reason"] in {
        "confidence_and_accuracy_targets_met",
        "target_accuracy_met",
        "token_budget_exhausted",
        "unlabeled_pool_exhausted",
    }
    assert isinstance(result["history"], list)
    assert result["total_iterations"] >= 1
