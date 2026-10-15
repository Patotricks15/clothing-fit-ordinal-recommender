"""Feature pipeline: builds a full feature matrix for any set of rows.

Static (user/item/context) features are read directly off each row and need
no fitting. The catalogue-wide item size summary (``fit_on: catalogue``) is
fit once on the full item catalogue. The historical aggregates are per-row
expanding statistics (see :mod:`recommender.features.interaction_features`)
computed once over the *entire* dataset (so that a user's or item's full,
already-observed history is available regardless of which split one of its
rows landed in) and then looked up per row by its unique
``(user_index, review_order)`` key -- unique because ``review_order`` is a
strictly increasing per-user counter assigned at load time
(:func:`recommender.data.loader.build_indexed_dataset`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data.loader import FitData
from recommender.features import (
    context_features,
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.registry import validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs

STATIC_COLUMNS = [
    "user_height_cm",
    "user_weight_kg",
    "user_bmi",
    "user_bust_band",
    "user_bust_cup_ordinal",
    "user_age",
    "item_nominal_size",
]
CATEGORICAL_COLUMNS = ["user_body_type", "item_category", "context_rented_for"]
HISTORICAL_COLUMNS = list(interaction_features.HISTORY_COLUMNS)
FEATURE_COLUMNS = STATIC_COLUMNS + CATEGORICAL_COLUMNS + HISTORICAL_COLUMNS


def assign_global_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a dataset-wide chronological order, used by the item-side cut-off."""
    frame = frame.sort_values(["review_date", "user_index", "item_index"], kind="mergesort")
    frame = frame.reset_index(drop=True)
    frame["global_order"] = np.arange(len(frame))
    return frame


class FeatureStore:
    """Fitted lookup tables plus the pure per-row static transforms.

    The same object featurizes training rows and a single incoming serving
    request, which is what makes offline/online parity testable rather than
    aspirational: both paths call :meth:`transform`.
    """

    def __init__(self, dataset: FitData, specs: list[FeatureSpec]) -> None:
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.specs = specs

        self.item_size_table = item_features.fit_item_nominal_size_table(
            dataset.frame, dataset.n_items
        )

        ordered = assign_global_order(dataset.frame)
        history = interaction_features.build_history_table(ordered)
        self._history_lookup = history.set_index(["user_index", "review_order"])[
            HISTORICAL_COLUMNS
        ]
        self.global_mismatch_rate = float(
            dataset.frame["fit"].isin(interaction_features.MISMATCH_LEVELS).mean()
        )
        self.categories_ = {
            col: sorted(dataset.frame[raw_col].dropna().unique().tolist())
            for col, raw_col in (
                ("user_body_type", "body_type"),
                ("item_category", "category"),
                ("context_rented_for", "rented_for"),
            )
        }

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Build the full feature matrix for an arbitrary set of rows.

        Args:
            rows: Any frame with the raw schema columns plus ``user_index``,
                ``item_index`` and ``review_order`` (the row's own position
                in its user's history, used only as the lookup key for the
                already-computed point-in-time history table).

        Returns:
            One row per input row, columns in :data:`FEATURE_COLUMNS` order,
            with every declared fill policy applied -- no nulls.
        """
        rows = rows.reset_index(drop=True)
        out: dict[str, pd.Series] = {
            "user_height_cm": user_features.user_height_cm(rows),
            "user_weight_kg": user_features.user_weight_kg(rows),
            "user_bmi": user_features.user_bmi(rows),
            "user_bust_band": user_features.user_bust_band(rows),
            "user_bust_cup_ordinal": user_features.user_bust_cup_ordinal(rows),
            "user_body_type": user_features.user_body_type(rows),
            "user_age": user_features.user_age(rows),
            "item_category": item_features.item_category(rows),
            "item_nominal_size": item_features.item_nominal_size(rows, self.item_size_table),
            "context_rented_for": context_features.context_rented_for(rows),
        }

        keys = pd.MultiIndex.from_arrays(
            [rows["user_index"].to_numpy(), rows["review_order"].to_numpy()]
        )
        history = self._history_lookup.reindex(keys).reset_index(drop=True)

        out["user_prior_review_count"] = history["user_prior_review_count"].fillna(0.0)
        out["user_prior_mismatch_rate"] = history["user_prior_mismatch_rate"].fillna(
            self.global_mismatch_rate
        )
        out["user_prior_small_rate"] = history["user_prior_small_rate"].fillna(0.0)
        out["item_prior_mismatch_rate"] = history["item_prior_mismatch_rate"].fillna(
            self.global_mismatch_rate
        )

        frame = pd.DataFrame(out)[FEATURE_COLUMNS]
        assert frame.isna().sum().sum() == 0, "feature matrix must have no nulls after fill"
        return frame

    def row_vector(self, row: pd.Series) -> dict[str, float | str]:
        """Featurize a single serving-time row (e.g. an API request)."""
        return self.transform(pd.DataFrame([row])).iloc[0].to_dict()

    def design_matrix_columns(self) -> list[str]:
        """Fixed column order of :meth:`encode`'s numeric design matrix."""
        columns = list(STATIC_COLUMNS) + list(HISTORICAL_COLUMNS)
        for col in CATEGORICAL_COLUMNS:
            for value in self.categories_[col]:
                columns.append(f"{col}={value}")
        return columns

    def encode(self, features: pd.DataFrame) -> np.ndarray:
        """One-hot encode categoricals (fixed category set) + numeric passthrough.

        Using the category set fitted from the whole dataset (not re-derived
        per call) keeps the design matrix's column count and meaning
        identical across train, validation, test and single-row serving
        requests -- an unseen category at serving time simply activates no
        one-hot column, rather than shifting every other column's index.

        Args:
            features: Output of :meth:`transform`.

        Returns:
            A dense ``float64`` array, columns ordered per
            :meth:`design_matrix_columns`.
        """
        numeric = features[STATIC_COLUMNS + HISTORICAL_COLUMNS].to_numpy(dtype=np.float64)
        blocks = [numeric]
        for col in CATEGORICAL_COLUMNS:
            values = features[col].to_numpy()
            one_hot = np.zeros((len(features), len(self.categories_[col])), dtype=np.float64)
            for j, category in enumerate(self.categories_[col]):
                one_hot[:, j] = values == category
            blocks.append(one_hot)
        return np.concatenate(blocks, axis=1)

    def to_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Convenience: featurize and append ``fit_code``/``fit`` for training."""
        features = self.transform(rows)
        features = features.reset_index(drop=True)
        features["fit_code"] = rows["fit_code"].reset_index(drop=True)
        features["fit"] = rows["fit"].reset_index(drop=True)
        return features


def build_features(dataset: FitData) -> FeatureStore:
    """Build the :class:`FeatureStore` for the whole dataset.

    Args:
        dataset: The full indexed :class:`~recommender.data.loader.FitData`.
            Static and catalogue features are fit on the whole catalogue by
            design (see ``conf/features.yaml``); historical aggregates use
            the whole dataset's chronology but strictly respect each row's
            own cut-off (see the module docstring).

    Returns:
        A populated :class:`FeatureStore`.
    """
    specs = load_feature_specs()
    validate_registry(specs)
    return FeatureStore(dataset, specs)
