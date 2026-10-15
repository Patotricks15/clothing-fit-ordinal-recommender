"""Baselines that the ordinal model must beat to justify its complexity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


@dataclass
class MajorityClassClassifier:
    """Always predicts the single most frequent training class.

    The floor every model must clear: on this fixture "fit" dominates (the
    real dataset is similarly skewed), so a model that cannot beat majority
    guessing is not using the features at all.
    """

    n_classes: int = 3
    majority_class_: int = 0
    class_prior_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> MajorityClassClassifier:  # noqa: ARG002
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        self.class_prior_ = counts / counts.sum()
        self.majority_class_ = int(np.argmax(counts))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the training class prior for every row (ignores ``X``)."""
        if self.class_prior_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        n = X.shape[0]
        return np.tile(self.class_prior_, (n, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.majority_class_)


@dataclass
class MultinomialLogisticRegression:
    """Plain (unordered) multinomial logistic regression.

    This is the baseline the ordinal model must beat on ordinal-aware
    metrics (mean absolute ordinal error, accuracy within one level): it
    uses the same features but treats "small"/"fit"/"large" as unordered
    categories, so it cannot exploit the ordering the way the proportional-
    odds model does. Trained with ``scipy.optimize`` as well, to keep the
    comparison purely about the ordinal structure, not the optimizer.
    """

    n_classes: int = 3
    l2: float = 1.0
    max_iter: int = 500
    weights_: np.ndarray | None = field(default=None, repr=False)
    n_features_: int = 0

    def _negative_log_likelihood(
        self, flat_weights: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        weights = flat_weights.reshape(self.n_classes, self.n_features_)
        logits = X @ weights.T
        probs = softmax(logits, axis=1)
        chosen = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
        nll = -np.log(chosen).sum()
        nll += 0.5 * self.l2 * float((weights**2).sum())
        return nll

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultinomialLogisticRegression:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_features_ = X.shape[1]

        x0 = np.zeros(self.n_classes * self.n_features_)
        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.weights_ = result.x.reshape(self.n_classes, self.n_features_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.weights_.T
        return softmax(logits, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
