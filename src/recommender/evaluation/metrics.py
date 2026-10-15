"""Ordinal evaluation metrics and conformal calibration diagnostics.

Standard ranking metrics (NDCG, Recall@K, MAP) do NOT apply to this project
and are deliberately absent from this module: those metrics score a *ranked
list* of items for a user from implicit feedback, but here every example is
an independent ordinal classification (what fit level, for this one
user-item pair) with a single, explicit ordered label -- there is no list to
rank and no notion of a top-K cut-off. The write-up
(``paper/sections/05-evaluation.tex``) states this explicitly so the absence
reads as a deliberate scope decision, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationReport:
    """Point-prediction and calibration metrics for one model on one split."""

    model: str
    accuracy: float
    mean_absolute_ordinal_error: float
    accuracy_within_one: float
    macro_f1: float
    n_samples: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "accuracy": round(self.accuracy, 4),
            "mae_ordinal": round(self.mean_absolute_ordinal_error, 4),
            "accuracy_within_1": round(self.accuracy_within_one, 4),
            "macro_f1": round(self.macro_f1, 4),
            "n_samples": self.n_samples,
        }


@dataclass(frozen=True)
class ConformalReport:
    """Calibration diagnostics for a :class:`SplitConformalClassifier`."""

    model: str
    target_coverage: float
    empirical_coverage: float
    average_set_size: float
    n_samples: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "target_coverage": round(self.target_coverage, 4),
            "empirical_coverage": round(self.empirical_coverage, 4),
            "average_set_size": round(self.average_set_size, 4),
            "n_samples": self.n_samples,
        }


def mean_absolute_ordinal_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean |true level - predicted level|, the natural ordinal error.

    Unlike plain accuracy, this penalizes a "small" predicted as "large"
    more than a "small" predicted as "fit" -- the distinction that motivates
    treating the problem as ordinal in the first place.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.abs(y_true.astype(np.int64) - y_pred.astype(np.int64)).mean())


def accuracy_within_one_level(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions at most one ordinal level away from the truth."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((np.abs(y_true.astype(np.int64) - y_pred.astype(np.int64)) <= 1).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1, robust to the "fit"-majority skew."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3, model_name: str = "model"
) -> ClassificationReport:
    """Compute the point-prediction report for one model on one partition."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return ClassificationReport(
        model=model_name,
        accuracy=float((y_true == y_pred).mean()),
        mean_absolute_ordinal_error=mean_absolute_ordinal_error(y_true, y_pred),
        accuracy_within_one=accuracy_within_one_level(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, n_classes),
        n_samples=len(y_true),
    )


def evaluate_conformal(
    conformal, probs: np.ndarray, y_true: np.ndarray, model_name: str = "model"  # noqa: ANN001
) -> ConformalReport:
    """Compute the calibration report for a fitted, calibrated conformal wrapper."""
    return ConformalReport(
        model=model_name,
        target_coverage=1.0 - conformal.alpha,
        empirical_coverage=conformal.coverage(probs, y_true),
        average_set_size=float(conformal.set_sizes(probs).mean()),
        n_samples=len(y_true),
    )
