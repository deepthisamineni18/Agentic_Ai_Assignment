"""
Active Learning Pipeline — Task 3
Agents: AnnotatorAgent, QualityAssessorAgent, TrainerAgent
Orchestrated by ActiveLearningPipeline.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from research_pipeline.llm_client import LLMClient
from research_pipeline.active_learning.datasets import get_active_learning_dataset

logger = logging.getLogger("ActiveLearning")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CLASSES = ["technology", "sports", "politics", "entertainment", "science"]

KEYWORDS: dict[str, list[str]] = {
    "technology": ["software", "hardware", "ai", "algorithm", "data", "computer",
                   "cloud", "network", "digital", "coding", "robot", "chip", "startup"],
    "sports":     ["football", "basketball", "soccer", "athlete", "game", "team",
                   "tournament", "championship", "league", "olympic", "score", "match"],
    "politics":   ["election", "government", "policy", "senate", "law", "president",
                   "vote", "congress", "democracy", "party", "campaign", "minister"],
    "entertainment": ["movie", "music", "celebrity", "award", "film", "actor",
                      "concert", "album", "streaming", "oscar", "performance", "theatre"],
    "science":    ["research", "discovery", "experiment", "biology", "physics",
                   "chemistry", "space", "genome", "telescope", "climate", "vaccine"],
}


def _generate_sample(label: str, seed: int) -> str:
    rng = random.Random(seed)
    core_words = KEYWORDS[label]
    other_words = ["the", "a", "is", "in", "of", "and", "to", "was", "has", "for",
                   "with", "on", "by", "from", "that", "this", "it", "are"]
    words: list[str] = []
    for _ in range(rng.randint(20, 60)):
        if rng.random() < 0.35:
            words.append(rng.choice(core_words))
        else:
            words.append(rng.choice(other_words))
    return " ".join(words)


def get_synthetic_dataset(num_samples: int = 500) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    per_class = num_samples // len(CLASSES)
    for cls_idx, cls in enumerate(CLASSES):
        for i in range(per_class):
            seed = cls_idx * 10_000 + i
            dataset.append({
                "id": f"{cls}_{i}",
                "text": _generate_sample(cls, seed),
                "true_label": cls,
                "label": None,
                "confidence": None,
            })
    random.shuffle(dataset)
    return dataset


# ---------------------------------------------------------------------------
# AnnotatorAgent
# ---------------------------------------------------------------------------

@dataclass
class AnnotationResult:
    sample_id: str
    text: str
    true_label: str
    label: str
    confidence: float


class AnnotatorAgent:
    """
    Labels samples with a real LLM call (Anthropic or Groq, whichever
    ANTHROPIC_API_KEY / GROQ_API_KEY is set — see llm_client.LLMClient).
    Uses a structured, few-shot classification prompt and asks the model to
    self-report a confidence score alongside the label.

    If no API key is configured (or a call fails for any reason — timeout,
    rate limit, malformed response), the annotator falls back to a
    keyword-density scorer so the pipeline still runs end-to-end offline,
    with no API key and no internet dependency, exactly like the rest of
    this repo's pipelines. Every fallback is logged so it's visible which
    labels came from the real model vs. the offline heuristic.

    Enforces a per-run token budget: real calls are charged their actual
    reported input+output token usage; fallback labeling is charged a fixed
    per-sample estimate (TOKENS_PER_SAMPLE_FALLBACK).
    Novel-sample selection via TF-IDF cosine distance.
    """

    TOKENS_PER_SAMPLE_FALLBACK = 300  # charged only when the LLM path isn't used
    # Used purely as an admission-control estimate before a real LLM call, since
    # actual usage isn't known until the response comes back.
    TOKENS_PER_SAMPLE_LLM_ESTIMATE = 400

    FEW_SHOT_EXAMPLES = [
        ("The senate voted on the new campaign finance bill after the president's speech.", "politics"),
        ("The startup unveiled a new AI chip that speeds up cloud computing workloads.", "technology"),
        ("The championship match went into overtime after both teams tied the score.", "sports"),
    ]

    def __init__(self, token_budget: int = 50_000, llm: LLMClient | None = None) -> None:
        self.token_budget = token_budget
        self.tokens_used = 0
        self.llm = llm if llm is not None else LLMClient()
        self.llm_calls = 0
        self.fallback_calls = 0
        if self.llm.is_available():
            logger.info("AnnotatorAgent: real LLM annotation enabled (model=%s).", self.llm.model)
        else:
            logger.info(
                "AnnotatorAgent: no ANTHROPIC_API_KEY/GROQ_API_KEY set — "
                "using offline keyword-density fallback for all annotations."
            )

    # ---- novelty-based selection ------------------------------------------

    def select_novel_samples(
        self,
        unlabeled_pool: list[dict[str, Any]],
        labeled_pool: list[dict[str, Any]],
        batch_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Returns the most novel (distant) unlabeled samples relative to labeled ones."""
        if not labeled_pool:
            return random.sample(unlabeled_pool, min(batch_size, len(unlabeled_pool)))

        labeled_texts   = [d["text"] for d in labeled_pool]
        unlabeled_texts = [d["text"] for d in unlabeled_pool]

        vec = TfidfVectorizer(max_features=3_000)
        vec.fit(labeled_texts + unlabeled_texts)

        X_lab   = vec.transform(labeled_texts)
        X_unlab = vec.transform(unlabeled_texts)

        # Maximum similarity of each unlabeled doc to any labeled doc
        sim = cosine_similarity(X_unlab, X_lab)
        max_sim = sim.max(axis=1)

        # Pick samples with the lowest max-similarity (most novel)
        idxs = np.argsort(max_sim)[:batch_size]
        selected = [unlabeled_pool[i] for i in idxs]
        logger.debug("Novelty selection: min_sim=%.4f  max_sim=%.4f",
                     max_sim[idxs[0]], max_sim[idxs[-1]])
        return selected

    # ---- prompting ----------------------------------------------------

    def _build_prompt(self, text: str) -> tuple[str, str]:
        """Structured zero/few-shot classification prompt. Instructs the
        model to reason silently but respond with ONLY machine-parseable
        JSON, which keeps token cost low and makes label+confidence
        extraction reliable."""
        examples = "\n".join(
            f'  Article: "{ex_text}"\n  -> {{"label": "{ex_label}", "confidence": 0.97}}'
            for ex_text, ex_label in self.FEW_SHOT_EXAMPLES
        )
        system = (
            "You are a precise text-classification annotator for a news-article "
            f"dataset. Classify each article into EXACTLY ONE of these categories: "
            f"{', '.join(CLASSES)}.\n\n"
            "Examples:\n" + examples + "\n\n"
            "Respond with ONLY a single-line JSON object and nothing else — no "
            "preamble, no markdown fences:\n"
            '{"label": "<one of the categories, lowercase>", "confidence": <float between 0.0 and 1.0>}\n'
            "The confidence field must reflect your genuine certainty that the "
            "label is correct, calibrated against how unambiguous the article's "
            "vocabulary and subject matter are."
        )
        user = f'Article:\n"""\n{text}\n"""\n\nClassify this article.'
        return system, user

    @staticmethod
    def _parse_llm_response(raw: str) -> tuple[str, float]:
        cleaned = raw.strip()
        # Strip markdown code fences if the model adds them despite instructions.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {raw!r}")
        parsed = json.loads(match.group(0))
        label = str(parsed["label"]).strip().lower()
        if label not in KEYWORDS:
            raise ValueError(f"LLM returned an unknown label: {label!r}")
        confidence = float(parsed["confidence"])
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return label, confidence

    # ---- keyword fallback ---------------------------------------------

    def _keyword_score(self, text: str) -> tuple[str, float]:
        """Offline heuristic used only when the LLM is unavailable or fails."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for cls, kws in KEYWORDS.items():
            scores[cls] = sum(1 for kw in kws if kw in text_lower)

        total = sum(scores.values())
        best  = max(scores, key=scores.get)

        if total == 0:
            confidence = 0.20
        else:
            raw_conf   = scores[best] / total
            confidence = float(np.clip(raw_conf * 0.9, 0.10, 0.99))
        return best, round(confidence, 3)

    # ---- annotation -----------------------------------------------------

    def annotate_sample(self, doc: dict[str, Any]) -> AnnotationResult | None:
        """Annotates one sample using a real LLM call when available, else the
        offline keyword fallback. Returns None when the token budget is exhausted."""
        estimate = (
            self.TOKENS_PER_SAMPLE_LLM_ESTIMATE
            if self.llm.is_available() else self.TOKENS_PER_SAMPLE_FALLBACK
        )
        if self.tokens_used + estimate > self.token_budget:
            logger.warning("Token budget exhausted — stopping annotation.")
            return None

        label: str | None = None
        confidence: float | None = None

        if self.llm.is_available():
            try:
                system, user = self._build_prompt(doc["text"])
                raw, in_tok, out_tok = self.llm.generate_with_usage(
                    system=system, user=user, max_tokens=60, temperature=0.0)
                label, confidence = self._parse_llm_response(raw)
                cost = in_tok + out_tok
                self.tokens_used += cost if cost > 0 else self.TOKENS_PER_SAMPLE_LLM_ESTIMATE
                self.llm_calls += 1
            except Exception as e:
                logger.warning(
                    "LLM annotation failed for sample %s (%s) — falling back to "
                    "keyword scorer for this sample.", doc["id"], e,
                )
                label = None  # fall through to the keyword path below

        if label is None:
            label, confidence = self._keyword_score(doc["text"])
            self.tokens_used += self.TOKENS_PER_SAMPLE_FALLBACK
            self.fallback_calls += 1

        return AnnotationResult(
            sample_id=doc["id"],
            text=doc["text"],
            true_label=doc["true_label"],
            label=label,
            confidence=round(confidence, 3),
        )

    def annotate_batch(
        self, batch: list[dict[str, Any]]
    ) -> tuple[list[AnnotationResult], bool]:
        """Annotates a batch. Returns (results, budget_hit)."""
        results: list[AnnotationResult] = []
        for doc in batch:
            r = self.annotate_sample(doc)
            if r is None:
                return results, True
            results.append(r)
        return results, False


# ---------------------------------------------------------------------------
# QualityAssessorAgent
# ---------------------------------------------------------------------------

class QualityAssessorAgent:
    """
    Reviews every annotation with confidence < threshold and gets a second
    opinion using a real LLM call (independent of the Annotator's own call,
    so it acts as a genuine review rather than trusting the same judgment
    twice). If the second opinion disagrees with the Annotator's label, it
    is re-assigned; either way confidence is adjusted based on whether the
    two opinions agreed.

    Falls back to a keyword-evidence re-check (no LLM call) when the LLM is
    unavailable or a call fails, so the pipeline still runs offline.
    """

    LOW_CONF_THRESHOLD = 0.55

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm if llm is not None else LLMClient()
        self.llm_calls = 0
        self.fallback_calls = 0

    def _build_review_prompt(self, text: str, proposed_label: str) -> tuple[str, str]:
        system = (
            "You are a quality-control reviewer for a news-article classification "
            f"dataset. The valid categories are: {', '.join(CLASSES)}. "
            "You will be shown an article and a proposed label from another "
            "annotator. Independently decide the correct label — do not simply "
            "defer to the proposed one. "
            "Respond with ONLY a single-line JSON object and nothing else:\n"
            '{"label": "<one of the categories, lowercase>", "confidence": <float 0.0-1.0>}'
        )
        user = (
            f'Article:\n"""\n{text}\n"""\n\n'
            f'Proposed label: "{proposed_label}"\n\n'
            "What is the correct label?"
        )
        return system, user

    def _keyword_recheck(self, text: str) -> tuple[str, int]:
        text_lower = text.lower()
        scores = {cls: sum(1 for kw in kws if kw in text_lower)
                  for cls, kws in KEYWORDS.items()}
        total = sum(scores.values())
        best  = max(scores, key=scores.get)
        return best, total

    def assess(self, annotations: list[AnnotationResult]) -> list[AnnotationResult]:
        reassigned = 0
        reviewed = 0
        for ann in annotations:
            if ann.confidence >= self.LOW_CONF_THRESHOLD:
                continue
            reviewed += 1

            best: str | None = None
            if self.llm.is_available():
                try:
                    system, user = self._build_review_prompt(ann.text, ann.label)
                    raw, _in_tok, _out_tok = self.llm.generate_with_usage(
                        system=system, user=user, max_tokens=60, temperature=0.0)
                    best, _llm_conf = AnnotatorAgent._parse_llm_response(raw)
                    self.llm_calls += 1
                except Exception as e:
                    logger.warning(
                        "QA LLM review failed for sample %s (%s) — falling back to "
                        "keyword re-check.", ann.sample_id, e,
                    )
                    best = None

            if best is None:
                keyword_best, total = self._keyword_recheck(ann.text)
                best = keyword_best if total > 0 else ann.label
                self.fallback_calls += 1

            if best != ann.label:
                old = ann.label
                ann.label = best
                ann.confidence = min(0.95, ann.confidence + 0.15)
                logger.debug("QA reassigned %s → %s (conf %.3f)", old, best, ann.confidence)
                reassigned += 1
            else:
                ann.confidence = min(0.95, ann.confidence + 0.10)

        logger.info("QualityAssessor: %d annotations reassigned out of %d low-conf items reviewed",
                    reassigned, reviewed)
        return annotations


# ---------------------------------------------------------------------------
# PyTorch LSTM
# ---------------------------------------------------------------------------

if torch is not None:
    class _LSTMClassifier(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.dropout   = nn.Dropout(0.3)
            self.fc        = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            emb = self.embedding(x)
            _, (hn, _) = self.lstm(emb)
            out = self.dropout(hn[-1])
            return self.fc(out)
else:
    class _LSTMClassifier:  # type: ignore[no-redef]
        pass


def _build_vocab(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for tok in text.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _texts_to_tensor(texts: list[str], vocab: dict[str, int], seq_len: int = 50) -> torch.Tensor:
    seqs = []
    for text in texts:
        ids = [vocab.get(t, 1) for t in text.lower().split()[:seq_len]]
        ids += [0] * (seq_len - len(ids))
        seqs.append(ids)
    return torch.tensor(seqs, dtype=torch.long)


# ---------------------------------------------------------------------------
# TrainerAgent
# ---------------------------------------------------------------------------

@dataclass
class TrainingReport:
    model_name: str
    val_accuracy: float
    test_accuracy: float
    metrics_per_class: dict[str, dict[str, float]]
    early_stopped: bool = False


class TrainerAgent:
    """
    Trains multiple classifiers on the labeled pool:
      - LogisticRegression  (sklearn, TF-IDF features)
      - RandomForest        (sklearn, TF-IDF features)
      - KNN                 (sklearn, TF-IDF features)
      - LSTM                (PyTorch)

    Selects the best by validation accuracy.
    Evaluates the winner on the held-out test split.
    Supports early stopping via target_accuracy.
    Reports per-class Precision / Recall / F1.
    """

    def __init__(self, target_accuracy: float = 0.85, max_lstm_epochs: int = 10) -> None:
        self.target_accuracy  = target_accuracy
        self.max_lstm_epochs  = max_lstm_epochs
        self.label_to_idx     = {cls: i for i, cls in enumerate(CLASSES)}
        self.idx_to_label     = {i: cls for i, cls in enumerate(CLASSES)}

    # ---- data preparation -------------------------------------------------

    def _split(
        self,
        annotations: list[AnnotationResult],
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        texts  = [a.text for a in annotations]
        labels = [a.true_label for a in annotations]

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            texts, labels, test_size=0.30, random_state=42, stratify=None)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42)
        return X_tr, X_val, X_te, y_tr, y_val, y_te

    # ---- sklearn models ---------------------------------------------------

    def _fit_sklearn(
        self,
        clf,
        X_tr, y_tr, X_val, y_val,
        vec: TfidfVectorizer,
    ) -> tuple[Any, float]:
        clf.fit(vec.transform(X_tr), y_tr)
        preds = clf.predict(vec.transform(X_val))
        acc   = accuracy_score(y_val, preds)
        return clf, acc

    # ---- LSTM -------------------------------------------------------------

    def _fit_lstm(
        self,
        X_tr, y_tr, X_val, y_val,
    ) -> tuple[tuple, float]:
        if torch is None or nn is None or optim is None or DataLoader is None or TensorDataset is None:
            logger.warning("PyTorch is unavailable; skipping LSTM training (%s).", TORCH_IMPORT_ERROR)
            return None, 0.0

        vocab  = _build_vocab(X_tr)
        seq_len = 50

        tr_X = _texts_to_tensor(X_tr, vocab, seq_len)
        va_X = _texts_to_tensor(X_val, vocab, seq_len)
        tr_y = torch.tensor([self.label_to_idx.get(l, 0) for l in y_tr], dtype=torch.long)
        va_y = torch.tensor([self.label_to_idx.get(l, 0) for l in y_val], dtype=torch.long)

        ds     = TensorDataset(tr_X, tr_y)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        model = _LSTMClassifier(
            vocab_size=len(vocab), embed_dim=32, hidden_dim=64,
            num_classes=len(CLASSES),
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val_acc   = 0.0
        early_stopped  = False

        for epoch in range(self.max_lstm_epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(va_X)
                preds  = logits.argmax(dim=1)
                val_acc = (preds == va_y).float().mean().item()

            logger.debug("LSTM epoch %d/%d  val_acc=%.4f", epoch + 1, self.max_lstm_epochs, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if best_val_acc >= self.target_accuracy:
                logger.info("LSTM early stop at epoch %d (val_acc=%.4f >= %.4f)",
                            epoch + 1, best_val_acc, self.target_accuracy)
                early_stopped = True
                break

        return (model, vocab, seq_len, early_stopped), best_val_acc

    # ---- evaluation -------------------------------------------------------

    def _evaluate_sklearn(self, clf, vec: TfidfVectorizer, X_te, y_te) -> tuple[list[str], float]:
        preds = clf.predict(vec.transform(X_te))
        return list(preds), accuracy_score(y_te, preds)

    def _evaluate_lstm(self, bundle: tuple, X_te, y_te) -> tuple[list[str], float]:
        model, vocab, seq_len, _ = bundle
        te_X = _texts_to_tensor(X_te, vocab, seq_len)
        te_y = torch.tensor([self.label_to_idx.get(l, 0) for l in y_te], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            preds_idx = model(te_X).argmax(dim=1).numpy()
        preds = [self.idx_to_label[i] for i in preds_idx]
        return preds, accuracy_score(y_te, preds)

    # ---- public API -------------------------------------------------------

    def train(self, annotations: list[AnnotationResult]) -> TrainingReport:
        """Trains all models and returns a TrainingReport for the best."""
        X_tr, X_val, X_te, y_tr, y_val, y_te = self._split(annotations)

        vec = TfidfVectorizer(max_features=5_000)
        vec.fit(X_tr)

        candidates: dict[str, tuple[Any, float]] = {}

        # sklearn
        for name, clf in [
            ("LogisticRegression", LogisticRegression(max_iter=300, random_state=42)),
            ("RandomForest",       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("KNN",                KNeighborsClassifier(n_neighbors=5)),
        ]:
            model, val_acc = self._fit_sklearn(clf, X_tr, y_tr, X_val, y_val, vec)
            candidates[name] = (model, val_acc)
            logger.info("%-22s  val_acc=%.4f", name, val_acc)

        # LSTM
        if torch is not None:
            lstm_bundle, lstm_val_acc = self._fit_lstm(X_tr, y_tr, X_val, y_val)
            candidates["LSTM"] = (lstm_bundle, lstm_val_acc)
            logger.info("%-22s  val_acc=%.4f", "LSTM", lstm_val_acc)
        else:
            logger.warning("PyTorch unavailable; using sklearn models only for this run.")

        # Select best
        best_name  = max(candidates, key=lambda k: candidates[k][1])
        best_model = candidates[best_name][0]
        best_val_acc = candidates[best_name][1]
        logger.info("Best model: %s  (val_acc=%.4f)", best_name, best_val_acc)

        # Evaluate on test split
        if best_name == "LSTM":
            test_preds, test_acc = self._evaluate_lstm(best_model, X_te, y_te)
            early_stopped = best_model[3]
        else:
            test_preds, test_acc = self._evaluate_sklearn(best_model, vec, X_te, y_te)
            early_stopped = False

        # Per-class metrics
        labels_present = sorted(set(y_te) | set(test_preds))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, test_preds, labels=labels_present, zero_division=0)
        metrics = {
            cls: {
                "precision": round(float(prec[i]), 3),
                "recall":    round(float(rec[i]),  3),
                "f1_score":  round(float(f1[i]),   3),
            }
            for i, cls in enumerate(labels_present)
        }

        logger.info("Test accuracy: %.4f  early_stopped=%s", test_acc, early_stopped)

        return TrainingReport(
            model_name=best_name,
            val_accuracy=round(best_val_acc, 4),
            test_accuracy=round(test_acc, 4),
            metrics_per_class=metrics,
            early_stopped=early_stopped,
        )


# ---------------------------------------------------------------------------
# ActiveLearningPipeline — orchestrator
# ---------------------------------------------------------------------------

class ActiveLearningPipeline:
    """
    Coordinates: novel selection → annotation → quality assessment → training.

    Two independent stopping criteria are checked every iteration, matching
    the two-stage spec (annotation pipeline vs. training pipeline):
      - Task 1 (annotation): stop once the mean confidence across the whole
        labeled pool reaches `annotation_confidence_target` (default 0.8).
      - Task 2 (training):   stop once the trained model's held-out test
        accuracy reaches `target_accuracy` (default 0.85).
    The loop also stops if the annotator's token budget is exhausted, or the
    unlabeled pool runs out.
    """

    def __init__(
        self,
        token_budget: int = 50_000,
        target_accuracy: float = 0.85,
        annotation_confidence_target: float = 0.8,
        batch_size: int = 20,
        num_samples: int = 500,
        use_real_dataset: bool = True,
    ) -> None:
        self.annotator = AnnotatorAgent(token_budget=token_budget)
        self.assessor  = QualityAssessorAgent()
        self.trainer   = TrainerAgent(target_accuracy=target_accuracy)
        self.batch_size     = batch_size
        self.target_accuracy = target_accuracy
        self.annotation_confidence_target = annotation_confidence_target
        self.num_samples    = num_samples
        self.use_real_dataset = use_real_dataset

    @staticmethod
    def _mean_confidence(labeled: list[AnnotationResult]) -> float:
        if not labeled:
            return 0.0
        return sum(a.confidence for a in labeled) / len(labeled)

    def run(self) -> dict[str, Any]:
        unlabeled = get_active_learning_dataset(
            use_real=self.use_real_dataset,
            num_samples=self.num_samples,
            synthetic_loader=get_synthetic_dataset,
        )
        # get_active_learning_dataset silently falls back to synthetic data
        # when AG News can't be loaded; sample ids are the only signal left
        # once we're back in this scope, so use that to report honestly
        # whether this was a real-data run rather than letting a fallback
        # run look identical to a real one in the summary.
        used_real_dataset = bool(unlabeled) and unlabeled[0]["id"].startswith("ag_")
        logger.info(
            "Dataset: %s",
            "AG News (real)" if used_real_dataset else "synthetic keyword-generated fallback",
        )

        labeled: list[AnnotationResult] = []

        # Seed set: first 20 samples labelled from ground truth (no token cost)
        seed_pool, unlabeled = unlabeled[:20], unlabeled[20:]
        for doc in seed_pool:
            labeled.append(AnnotationResult(
                sample_id=doc["id"],
                text=doc["text"],
                true_label=doc["true_label"],
                label=doc["true_label"],
                confidence=1.0,
            ))
        logger.info("Seed set: %d samples.", len(labeled))

        history: list[dict[str, Any]] = []
        iteration = 0
        stop_reason = "unlabeled_pool_exhausted"

        while len(unlabeled) >= self.batch_size:
            iteration += 1
            logger.info("=== Iteration %d  (labeled=%d  unlabeled=%d) ===",
                        iteration, len(labeled), len(unlabeled))

            # 1. Select novel samples
            labeled_dicts = [{"id": a.sample_id, "text": a.text} for a in labeled]
            batch = self.annotator.select_novel_samples(
                unlabeled, labeled_dicts, self.batch_size)

            # 2. Annotate
            annotated, budget_hit = self.annotator.annotate_batch(batch)
            if not annotated:
                logger.info("Budget exhausted before any annotations — stopping.")
                stop_reason = "token_budget_exhausted"
                break

            # 3. Quality assessment
            assessed = self.assessor.assess(annotated)

            # 4. Update pools
            annotated_ids = {a.sample_id for a in assessed}
            unlabeled = [d for d in unlabeled if d["id"] not in annotated_ids]
            labeled.extend(assessed)

            mean_conf = self._mean_confidence(labeled)

            # 5. Train
            report = self.trainer.train(labeled)
            logger.info("Iteration %d  best=%s  test_acc=%.4f  mean_annotation_conf=%.3f",
                        iteration, report.model_name, report.test_accuracy, mean_conf)

            history.append({
                "iteration":            iteration,
                "labeled_count":        len(labeled),
                "mean_confidence":      round(mean_conf, 4),
                "model_name":           report.model_name,
                "val_accuracy":         report.val_accuracy,
                "test_accuracy":        report.test_accuracy,
                "tokens_used":          self.annotator.tokens_used,
                "llm_calls":            self.annotator.llm_calls + self.assessor.llm_calls,
                "fallback_calls":       self.annotator.fallback_calls + self.assessor.fallback_calls,
                "early_stopped":        report.early_stopped,
            })

            if budget_hit:
                logger.info("Token budget hit — stopping.")
                stop_reason = "token_budget_exhausted"
                break
            if mean_conf >= self.annotation_confidence_target and report.test_accuracy >= self.target_accuracy:
                logger.info(
                    "Annotation confidence %.3f >= %.2f AND test accuracy %.4f >= %.2f — stopping.",
                    mean_conf, self.annotation_confidence_target, report.test_accuracy, self.target_accuracy,
                )
                stop_reason = "confidence_and_accuracy_targets_met"
                break
            if report.test_accuracy >= self.target_accuracy:
                logger.info("Target accuracy %.2f reached — stopping.", self.target_accuracy)
                stop_reason = "target_accuracy_met"
                break

        return {
            "total_iterations":  iteration,
            "labeled_count":     len(labeled),
            "tokens_used":       self.annotator.tokens_used,
            "llm_calls":         self.annotator.llm_calls + self.assessor.llm_calls,
            "fallback_calls":    self.annotator.fallback_calls + self.assessor.fallback_calls,
            "mean_confidence":   round(self._mean_confidence(labeled), 4),
            "stop_reason":       stop_reason,
            "used_real_dataset": used_real_dataset,
            "used_real_llm":     self.annotator.llm.is_available(),
            "history":           history,
            "final_report": {
                "model_name":        history[-1]["model_name"]       if history else None,
                "test_accuracy":     history[-1]["test_accuracy"]    if history else 0.0,
                "metrics_per_class": self.trainer.train(labeled).metrics_per_class if labeled else {},
            },
        }
