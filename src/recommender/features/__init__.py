"""Point-in-time, declaratively-specified feature engineering."""

from __future__ import annotations

from recommender.features.pipeline import FEATURE_COLUMNS, FeatureStore, build_features
from recommender.features.registry import TRANSFORM_REGISTRY, validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs

__all__ = [
    "FEATURE_COLUMNS",
    "FeatureSpec",
    "FeatureStore",
    "TRANSFORM_REGISTRY",
    "build_features",
    "load_feature_specs",
    "validate_registry",
]
