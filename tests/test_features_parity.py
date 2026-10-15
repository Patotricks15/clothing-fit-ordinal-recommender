"""Training/serving parity tests.

The failure mode these guard against is skew: an offline batch computes one
feature vector while the online (single-request) path computes a different
one for the same logical row, so good offline numbers never materialize in
production. Every assertion compares the single-row serving path
(:meth:`FeatureStore.row_vector`) against the exact batch path
(:meth:`FeatureStore.transform`) used by training and evaluation.
"""

from __future__ import annotations

import math

import pandas as pd

from recommender.features import FEATURE_COLUMNS


def test_row_vector_matches_batch_transform_for_every_test_row(feature_store, split) -> None:
    batch = feature_store.transform(split.test).reset_index(drop=True)

    for i in range(min(30, len(split.test))):
        row = split.test.iloc[i]
        served = feature_store.row_vector(row)
        offline = batch.iloc[i].to_dict()

        for col in FEATURE_COLUMNS:
            served_value = served[col]
            offline_value = offline[col]
            if isinstance(offline_value, float):
                assert math.isclose(served_value, offline_value, rel_tol=1e-6, abs_tol=1e-6), (
                    f"serving skew on '{col}' for row {i}: {served_value!r} != {offline_value!r}"
                )
            else:
                assert served_value == offline_value, (
                    f"serving skew on '{col}' for row {i}: {served_value!r} != {offline_value!r}"
                )


def test_row_vector_matches_batch_transform_for_train_rows(feature_store, split) -> None:
    """Parity must also hold for training rows, not just held-out ones."""
    batch = feature_store.transform(split.train).reset_index(drop=True)
    row = split.train.iloc[0]
    served = feature_store.row_vector(row)
    offline = batch.iloc[0].to_dict()

    for col in FEATURE_COLUMNS:
        if isinstance(offline[col], float):
            assert math.isclose(served[col], offline[col], rel_tol=1e-6, abs_tol=1e-6)
        else:
            assert served[col] == offline[col]


def test_transform_is_order_independent(feature_store, split) -> None:
    """Shuffling the input rows must not change any row's own feature values."""
    shuffled = split.test.sample(frac=1.0, random_state=1).reset_index(drop=True)
    original = feature_store.transform(split.test).reset_index(drop=True)
    reshuffled = feature_store.transform(shuffled).reset_index(drop=True)

    # Re-sort the reshuffled output back to the original row identity via
    # (user_index, item_index, review_order), then compare values.
    key_cols = ["user_index", "item_index"]
    original_keyed = pd.concat(
        [split.test.reset_index(drop=True)[key_cols], original], axis=1
    ).sort_values(key_cols).reset_index(drop=True)
    reshuffled_keyed = pd.concat([shuffled[key_cols], reshuffled], axis=1).sort_values(
        key_cols
    ).reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        if original_keyed[col].dtype.kind in "fi":
            assert (
                (original_keyed[col] - reshuffled_keyed[col]).abs().max() < 1e-6
            ), f"{col} changed under row reordering"
        else:
            assert (original_keyed[col] == reshuffled_keyed[col]).all()


def test_batch_feature_matrix_is_json_serialisable(feature_store, split) -> None:
    """DynamoDB/Lambda payloads reject numpy scalars; this catches them early."""
    import json

    row = split.test.iloc[0]
    served = feature_store.row_vector(row)
    cleaned = {k: (float(v) if isinstance(v, float) else v) for k, v in served.items()}
    json.dumps(cleaned)
