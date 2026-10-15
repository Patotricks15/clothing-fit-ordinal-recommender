"""Build DynamoDB-shaped precomputed prediction records.

Design note: unlike a top-N ranking service, a clothing-fit predictor scores
a single ``(user, item, requested size)`` request, not a list. We therefore
precompute, for every known request we can anticipate (here: every row in a
partition), the full calibrated answer -- point prediction, class
probabilities, and the conformal prediction set -- and store it keyed by a
composite request id. The inference Lambda is then a pure key lookup, with
the same latency-boundedness rationale as a precomputed top-N table: the
model's forward pass is cheap, but doing it inside the request path would
still mean shipping the (small) fitted model and feature store into every
Lambda container.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recommender.data.loader import FIT_LEVELS
from recommender.features.pipeline import FeatureStore
from recommender.models.conformal import SplitConformalClassifier
from recommender.models.ordinal import OrdinalLogisticRegression
from recommender.models.preprocessing import Standardizer


def request_id(user_id: int, item_id: int, size: int) -> str:
    """Composite key identifying one fit-prediction request."""
    return f"{user_id}#{item_id}#{size}"


def build_prediction_payloads(
    model: OrdinalLogisticRegression,
    scaler: Standardizer,
    conformal: SplitConformalClassifier,
    feature_store: FeatureStore,
    rows: pd.DataFrame,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> list[dict[str, Any]]:
    """Precompute one calibrated prediction payload per row.

    Args:
        model: Fitted :class:`OrdinalLogisticRegression`.
        scaler: Fitted :class:`Standardizer` matching the model's training.
        conformal: Calibrated :class:`SplitConformalClassifier`.
        feature_store: Fitted feature store used both offline and online.
        rows: The rows to precompute payloads for (e.g. the test partition).
        user_ids: Original user ids, positionally aligned to ``user_index``.
        item_ids: Original item ids, positionally aligned to ``item_index``.

    Returns:
        One JSON-serialisable payload per row, in DynamoDB item shape.

    Raises:
        ValueError: If ``user_ids``/``item_ids`` cannot cover every index in
            ``rows``.
    """
    if rows["user_index"].max() >= len(user_ids):
        raise ValueError("user_ids is misaligned with rows['user_index'].")
    if rows["item_index"].max() >= len(item_ids):
        raise ValueError("item_ids is misaligned with rows['item_index'].")

    features = feature_store.transform(rows)
    X = scaler.transform(feature_store.encode(features))
    probs = model.predict_proba(X)
    predicted = probs.argmax(axis=1)
    sets = conformal.predict_sets(probs)

    payloads = []
    for i in range(len(rows)):
        row = rows.iloc[i]
        uid = int(user_ids[int(row["user_index"])])
        iid = int(item_ids[int(row["item_index"])])
        size = int(row["size"])

        payloads.append(
            {
                "request_id": request_id(uid, iid, size),
                "user_id": str(uid),
                "item_id": str(iid),
                "size": size,
                "model": "ordinal_logistic_regression",
                "predicted_fit": FIT_LEVELS[int(predicted[i])],
                "probabilities": {
                    level: round(float(probs[i, k]), 4) for k, level in enumerate(FIT_LEVELS)
                },
                "conformal_set": [
                    level for k, level in enumerate(FIT_LEVELS) if sets[i, k]
                ],
                "conformal_alpha": conformal.alpha,
            }
        )
    return payloads
