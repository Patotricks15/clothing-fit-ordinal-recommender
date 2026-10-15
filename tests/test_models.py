"""Tests for the ordinal model, baselines, conformal wrapper and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from recommender.evaluation import (
    accuracy_within_one_level,
    evaluate_classification,
    evaluate_conformal,
    macro_f1,
    mean_absolute_ordinal_error,
)
from recommender.models import (
    MajorityClassClassifier,
    MultinomialLogisticRegression,
    OrdinalLogisticRegression,
    SplitConformalClassifier,
    Standardizer,
)


@pytest.fixture(scope="module")
def design_matrices(feature_store, split):
    train_X = feature_store.encode(feature_store.transform(split.train))
    val_X = feature_store.encode(feature_store.transform(split.validation))
    test_X = feature_store.encode(feature_store.transform(split.test))
    scaler = Standardizer().fit(train_X)
    return {
        "train": (scaler.transform(train_X), split.train["fit_code"].to_numpy()),
        "validation": (scaler.transform(val_X), split.validation["fit_code"].to_numpy()),
        "test": (scaler.transform(test_X), split.test["fit_code"].to_numpy()),
    }


@pytest.fixture(scope="module")
def fitted_ordinal(design_matrices):
    X, y = design_matrices["train"]
    return OrdinalLogisticRegression(l2=2.0).fit(X, y)


# --- OrdinalLogisticRegression ----------------------------------------------


def test_ordinal_model_converges(fitted_ordinal) -> None:
    assert fitted_ordinal.converged_


def test_ordinal_thresholds_are_strictly_increasing(fitted_ordinal) -> None:
    thresholds = fitted_ordinal.thresholds_
    assert (np.diff(thresholds) > 0).all()


def test_ordinal_probabilities_sum_to_one(design_matrices, fitted_ordinal) -> None:
    X, _ = design_matrices["test"]
    probs = fitted_ordinal.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs >= 0).all()


def test_ordinal_rejects_out_of_range_labels(design_matrices) -> None:
    X, _ = design_matrices["train"]
    with pytest.raises(ValueError, match="Labels must be"):
        OrdinalLogisticRegression(n_classes=3).fit(X, np.full(len(X), 5))


def test_ordinal_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        OrdinalLogisticRegression().predict(np.zeros((2, 3)))


def test_ordinal_beats_majority_baseline_on_accuracy(design_matrices, fitted_ordinal) -> None:
    """The headline model must earn its complexity against the naive floor."""
    X, y = design_matrices["test"]
    train_X, train_y = design_matrices["train"]

    majority = MajorityClassClassifier().fit(train_X, train_y)
    ordinal_acc = (fitted_ordinal.predict(X) == y).mean()
    majority_acc = (majority.predict(X) == y).mean()

    assert ordinal_acc >= majority_acc


def test_ordinal_beats_multinomial_on_ordinal_error(design_matrices, fitted_ordinal) -> None:
    """The proportional-odds structure should pay off on the ordinal-aware metric."""
    X, y = design_matrices["test"]
    train_X, train_y = design_matrices["train"]

    mlr = MultinomialLogisticRegression(l2=2.0).fit(train_X, train_y)
    ordinal_mae = mean_absolute_ordinal_error(y, fitted_ordinal.predict(X))
    mlr_mae = mean_absolute_ordinal_error(y, mlr.predict(X))

    assert ordinal_mae <= mlr_mae + 1e-6


# --- Baselines ---------------------------------------------------------------


def test_majority_classifier_predicts_the_training_mode(design_matrices) -> None:
    X, y = design_matrices["train"]
    model = MajorityClassClassifier().fit(X, y)
    expected = int(np.bincount(y).argmax())
    assert model.majority_class_ == expected
    assert (model.predict(X) == expected).all()


def test_multinomial_probabilities_sum_to_one(design_matrices) -> None:
    X, y = design_matrices["train"]
    model = MultinomialLogisticRegression(l2=1.0).fit(X, y)
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# --- Split conformal prediction ----------------------------------------------


def test_conformal_achieves_approximate_target_coverage(design_matrices, fitted_ordinal) -> None:
    val_X, val_y = design_matrices["validation"]
    test_X, test_y = design_matrices["test"]

    conformal = SplitConformalClassifier(alpha=0.1).calibrate(
        fitted_ordinal.predict_proba(val_X), val_y
    )
    coverage = conformal.coverage(fitted_ordinal.predict_proba(test_X), test_y)

    # Finite-sample split conformal guarantees marginal coverage >= 1 - alpha
    # up to O(1/n) slack; a generous tolerance keeps the test from being flaky
    # on a modest synthetic calibration set while still catching gross breaks.
    assert coverage >= 0.9 - 0.08


def test_conformal_set_sizes_are_between_zero_and_n_classes(
    design_matrices, fitted_ordinal
) -> None:
    test_X, _ = design_matrices["test"]
    val_X, val_y = design_matrices["validation"]
    conformal = SplitConformalClassifier(alpha=0.1).calibrate(
        fitted_ordinal.predict_proba(val_X), val_y
    )
    sizes = conformal.set_sizes(fitted_ordinal.predict_proba(test_X))
    assert (sizes >= 0).all()
    assert (sizes <= 3).all()


def test_conformal_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        SplitConformalClassifier(alpha=1.5).calibrate(np.ones((3, 3)) / 3, np.zeros(3, dtype=int))


def test_conformal_predict_sets_requires_calibration() -> None:
    with pytest.raises(RuntimeError):
        SplitConformalClassifier().predict_sets(np.ones((2, 3)) / 3)


def test_tighter_alpha_never_shrinks_the_prediction_set(design_matrices, fitted_ordinal) -> None:
    """A stricter coverage target must not produce smaller (more confident) sets."""
    val_X, val_y = design_matrices["validation"]
    test_X, _ = design_matrices["test"]

    loose = SplitConformalClassifier(alpha=0.3).calibrate(
        fitted_ordinal.predict_proba(val_X), val_y
    )
    tight = SplitConformalClassifier(alpha=0.05).calibrate(
        fitted_ordinal.predict_proba(val_X), val_y
    )

    probs = fitted_ordinal.predict_proba(test_X)
    assert tight.set_sizes(probs).mean() >= loose.set_sizes(probs).mean()


# --- Evaluation metrics -------------------------------------------------------


def test_mean_absolute_ordinal_error_matches_manual_computation() -> None:
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 2, 0, 2])
    assert mean_absolute_ordinal_error(y_true, y_pred) == pytest.approx((0 + 1 + 2 + 2) / 4)


def test_accuracy_within_one_level() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 1, 0])
    assert accuracy_within_one_level(y_true, y_pred) == pytest.approx(2 / 3)


def test_macro_f1_perfect_predictions_is_one() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y, n_classes=3) == pytest.approx(1.0)


def test_evaluate_classification_report_fields(design_matrices, fitted_ordinal) -> None:
    X, y = design_matrices["test"]
    report = evaluate_classification(
        y, fitted_ordinal.predict(X), n_classes=3, model_name="OrdinalLR"
    )
    row = report.as_row()
    assert row["model"] == "OrdinalLR"
    assert 0.0 <= row["accuracy"] <= 1.0
    assert row["n_samples"] == len(y)


def test_evaluate_conformal_report_fields(design_matrices, fitted_ordinal) -> None:
    val_X, val_y = design_matrices["validation"]
    test_X, test_y = design_matrices["test"]
    conformal = SplitConformalClassifier(alpha=0.1).calibrate(
        fitted_ordinal.predict_proba(val_X), val_y
    )
    report = evaluate_conformal(
        conformal, fitted_ordinal.predict_proba(test_X), test_y, model_name="OrdinalLR"
    )
    assert report.target_coverage == pytest.approx(0.9)
    assert report.n_samples == len(test_y)
