"""Schema contract tests for the declarative feature specification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.features import FEATURE_COLUMNS, load_feature_specs, validate_registry
from recommender.features.spec import VALID_FAMILIES, FeatureSpec


def test_registry_matches_declaration() -> None:
    """Every declared feature is implemented and vice versa."""
    validate_registry(load_feature_specs())


def test_every_feature_declares_a_known_family() -> None:
    for spec in load_feature_specs():
        assert spec.family in VALID_FAMILIES


def test_every_feature_documents_a_rationale() -> None:
    """A feature without a stated reason cannot be reviewed or ablated."""
    for spec in load_feature_specs():
        assert len(spec.rationale.strip()) > 20, f"{spec.name} has a thin rationale"


def test_interaction_features_declare_point_in_time() -> None:
    """Every aggregate over review history must declare the cut-off contract."""
    for spec in load_feature_specs():
        if spec.family == "interaction":
            assert spec.point_in_time, f"{spec.name} aggregates history but is not point-in-time"


def test_static_user_and_item_features_need_no_cutoff() -> None:
    """Static attributes are known before any review and need no cut-off."""
    for spec in load_feature_specs():
        if spec.name in {
            "user_height_cm", "user_weight_kg", "user_bmi", "user_bust_band",
            "user_bust_cup_ordinal", "user_body_type", "user_age", "item_category",
        }:
            assert not spec.point_in_time, f"{spec.name} is static but declares point_in_time"


def test_catalogue_features_are_not_point_in_time() -> None:
    for spec in load_feature_specs():
        if spec.fit_on == "catalogue":
            assert not spec.point_in_time


def test_invalid_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        FeatureSpec(
            name="bad", family="nonsense", dtype="float32", transform="t",
            params={}, fit_on="none", point_in_time=False, fill={},
            rationale="long enough rationale for the validator to accept it",
        )


def test_missing_rationale_is_rejected() -> None:
    with pytest.raises(ValueError, match="rationale"):
        FeatureSpec(
            name="bad", family="item", dtype="float32", transform="t",
            params={}, fit_on="catalogue", point_in_time=False, fill={},
            rationale="   ",
        )


def test_interaction_feature_missing_point_in_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="point_in_time"):
        FeatureSpec(
            name="bad", family="interaction", dtype="float32", transform="t",
            params={}, fit_on="none", point_in_time=False, fill={},
            rationale="long enough rationale for the validator to accept it",
        )


def test_feature_store_has_no_nulls_and_correct_shape(feature_store, split) -> None:
    """Declared fill policies must leave no missing values downstream."""
    features = feature_store.transform(split.train)

    assert features.isna().sum().sum() == 0
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(split.train)


def test_numeric_features_are_finite(feature_store, split) -> None:
    features = feature_store.transform(split.test)
    numeric_cols = [c for c in FEATURE_COLUMNS if features[c].dtype.kind in "fi"]
    for col in numeric_cols:
        assert np.isfinite(features[col].to_numpy()).all(), f"{col} has non-finite values"


def test_rates_are_within_unit_interval(feature_store, split) -> None:
    features = feature_store.transform(split.validation)
    for col in ("user_prior_mismatch_rate", "user_prior_small_rate", "item_prior_mismatch_rate"):
        assert features[col].between(0.0, 1.0).all(), f"{col} escaped [0, 1]"


def test_unknown_item_falls_back_to_declared_defaults(feature_store, dataset) -> None:
    """Cold items must not raise; they must return the declared fill value."""
    from recommender.features import item_features

    ghost_item_index = dataset.n_items + 999
    ghost_row = pd.DataFrame({"item_index": [ghost_item_index]})
    size = item_features.item_nominal_size(ghost_row, feature_store.item_size_table).iloc[0]
    assert np.isfinite(size)
