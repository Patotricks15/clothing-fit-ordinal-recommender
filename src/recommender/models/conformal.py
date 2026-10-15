"""Split conformal prediction for ordinal/multiclass classifiers.

Implemented from scratch in numpy, on top of any fitted classifier exposing
``predict_proba``. See ``paper/sections/04-methodology.tex`` Equations (4)-(5)
for the derivation; this module implements the nonconformity score and the
split-conformal calibration procedure described there.

Nonconformity score
--------------------
For a calibration example ``(x_i, y_i)`` with fitted class probabilities
``p_hat(. | x_i)``, the score is

    alpha_i = 1 - p_hat(y_i | x_i)

-- large when the model placed little probability mass on the true label.

Calibration
-----------
Given a miscoverage level ``alpha`` and ``n`` calibration examples, let
``q_hat`` be the ``ceil((n + 1) * (1 - alpha)) / n``-th empirical quantile of
``{alpha_1, ..., alpha_n}``. For a new point ``x``, the conformal prediction
set is

    C(x) = { y : 1 - p_hat(y | x) <= q_hat }

Under the sole assumption that the calibration and test points are
exchangeable, this construction guarantees the marginal coverage bound

    P(y in C(x)) >= 1 - alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitConformalClassifier:
    """Split conformal wrapper calibrated on a held-out partition.

    Attributes:
        alpha: Target miscoverage rate (``1 - alpha`` is the coverage goal).
        q_hat: Fitted conformal quantile, set after :meth:`calibrate`.
    """

    alpha: float = 0.1
    q_hat: float | None = None
    n_calibration_: int = 0

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray) -> SplitConformalClassifier:
        """Fit ``q_hat`` on a calibration set disjoint from training.

        Args:
            probs: Fitted ``predict_proba`` output on the calibration
                partition, shape ``(n_cal, n_classes)``.
            y_true: True labels on the calibration partition.

        Returns:
            ``self``, with ``q_hat`` populated.

        Raises:
            ValueError: If ``alpha`` is outside ``(0, 1)`` or the
                calibration set is empty.
        """
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}.")
        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set must be non-empty.")

        scores = 1.0 - probs[np.arange(n), y_true]
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        self.n_calibration_ = n
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Boolean inclusion matrix, shape ``(n_samples, n_classes)``.

        Args:
            probs: Fitted ``predict_proba`` output on new data.

        Returns:
            ``included[i, c]`` is ``True`` iff class ``c`` belongs to the
            conformal prediction set for row ``i``.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        if self.q_hat is None:
            raise RuntimeError("Conformal predictor is not calibrated; call calibrate() first.")
        return (1.0 - probs) <= self.q_hat

    def set_sizes(self, probs: np.ndarray) -> np.ndarray:
        """Number of classes included per row."""
        return self.predict_sets(probs).sum(axis=1)

    def coverage(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Empirical fraction of rows whose true label is in its set."""
        included = self.predict_sets(probs)
        n = len(y_true)
        return float(included[np.arange(n), y_true].mean())
