"""Feature standardization fit on the training partition only.

Kept separate from :mod:`recommender.features` because this is a model
input transform (the same declared feature can be fed to models that do or
do not need scaling), not a declared, point-in-time-sensitive feature in its
own right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Standardizer:
    """Zero-mean, unit-variance scaling with train-only fitted statistics."""

    mean_: np.ndarray | None = field(default=None, repr=False)
    scale_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted; call fit() first.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
