"""Ordinal logistic regression (proportional-odds / cumulative-link model).

Classical machine learning only: parameters are fit by maximizing the
likelihood with :func:`scipy.optimize.minimize`, not by any neural-network
training loop. See ``paper/sections/04-methodology.tex`` for the full
derivation; this module is the implementation of Equations (1)-(3) there.

Model
-----
For an ordinal outcome with ``K`` ordered levels (here ``K=3``:
small < fit < large) and feature vector ``x``, the cumulative-link model
with a logistic link assumes

    P(y <= k | x) = sigma(theta_k - w^T x),  k = 1, ..., K-1

with ordered thresholds theta_1 < theta_2 < ... < theta_{K-1} and
sigma(z) = 1 / (1 + exp(-z)). The per-class probability is the first
difference of this cumulative distribution,
``P(y = c | x) = P(y <= c | x) - P(y <= c-1 | x)``, with the conventions
``P(y <= -1 | x) = 0`` and ``P(y <= K-1 | x) = 1``.

We use a single global weight vector ``w`` shared across thresholds (the
"parallel lines" / proportional-odds assumption): a one-unit increase in any
feature shifts every cumulative log-odds by the same amount, which is what
makes the model ordinal rather than an unordered multinomial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid


def _softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(z))``."""
    return np.logaddexp(0.0, z)


def _thresholds_from_raw(raw_thresholds: np.ndarray) -> np.ndarray:
    """Map an unconstrained vector to a strictly increasing threshold vector.

    ``theta_1 = raw[0]`` and every subsequent gap is a softplus of the
    corresponding raw parameter, so ``theta_k = theta_1 + sum(softplus(raw[1:k]))``
    is guaranteed strictly increasing for any real-valued ``raw``. This lets
    an unconstrained optimizer (:func:`scipy.optimize.minimize` with BFGS)
    search over threshold vectors without ever producing an invalid,
    non-monotonic set of cut-points.
    """
    gaps = np.concatenate([raw_thresholds[:1], _softplus(raw_thresholds[1:])])
    return np.cumsum(gaps)


def _class_probabilities(thresholds: np.ndarray, linear: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class probabilities for every row, shape ``(n_rows, n_classes)``."""
    n_rows = linear.shape[0]
    cum = np.empty((n_rows, n_classes + 1))
    cum[:, 0] = 0.0
    cum[:, -1] = 1.0
    for k in range(n_classes - 1):
        cum[:, k + 1] = sigmoid(thresholds[k] - linear)
    probs = np.diff(cum, axis=1)
    return np.clip(probs, 1e-12, 1.0)


@dataclass
class OrdinalLogisticRegression:
    """Proportional-odds ordinal classifier trained with ``scipy.optimize``.

    Attributes:
        n_classes: Number of ordered levels (3 for small/fit/large).
        l2: L2 penalty strength on the weight vector (thresholds are never
            penalized, since shrinking them has no natural interpretation).
        weights_: Fitted weight vector, set after :meth:`fit`.
        thresholds_: Fitted, strictly increasing cut-points, set after
            :meth:`fit`.
    """

    n_classes: int = 3
    l2: float = 1.0
    max_iter: int = 500
    weights_: np.ndarray | None = field(default=None, repr=False)
    thresholds_: np.ndarray | None = field(default=None, repr=False)
    n_features_: int = 0
    converged_: bool = False

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_thresholds = params[: self.n_classes - 1]
        weights = params[self.n_classes - 1 :]
        return _thresholds_from_raw(raw_thresholds), weights

    def _negative_log_likelihood(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        thresholds, weights = self._unpack(params)
        linear = X @ weights
        probs = _class_probabilities(thresholds, linear, self.n_classes)
        chosen = probs[np.arange(len(y)), y]
        nll = -np.log(chosen).sum()
        nll += 0.5 * self.l2 * float(weights @ weights)
        return nll

    def fit(self, X: np.ndarray, y: np.ndarray) -> OrdinalLogisticRegression:
        """Fit thresholds and weights by minimizing the penalized NLL.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.
            y: Integer class labels in ``{0, ..., n_classes - 1}``.

        Returns:
            ``self``, with ``weights_``/``thresholds_`` populated.

        Raises:
            ValueError: If ``y`` contains a label outside the declared range.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(
                f"Labels must be in [0, {self.n_classes - 1}], got range "
                f"[{y.min()}, {y.max()}]."
            )

        self.n_features_ = X.shape[1]
        # Initialize thresholds spread around zero and weights at zero.
        init_thresholds = np.linspace(-1.0, 1.0, self.n_classes - 1)
        init_weights = np.zeros(self.n_features_)
        x0 = np.concatenate([init_thresholds, init_weights])

        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )

        self.thresholds_, self.weights_ = self._unpack(result.x)
        self.converged_ = bool(result.success)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-class probabilities, shape ``(n_samples, n_classes)``."""
        if self.weights_ is None or self.thresholds_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        linear = X @ self.weights_
        return _class_probabilities(self.thresholds_, linear, self.n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class predictions (argmax over :meth:`predict_proba`)."""
        return self.predict_proba(X).argmax(axis=1)
