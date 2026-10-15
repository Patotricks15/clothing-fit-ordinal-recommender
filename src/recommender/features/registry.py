"""Registry binding declared feature names to their implementations."""

from __future__ import annotations

from collections.abc import Callable

from recommender.features import (
    context_features,
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.spec import FeatureSpec

TRANSFORM_REGISTRY: dict[str, Callable[..., object]] = {
    "user_height_cm": user_features.user_height_cm,
    "user_weight_kg": user_features.user_weight_kg,
    "user_bmi": user_features.user_bmi,
    "user_bust_band": user_features.user_bust_band,
    "user_bust_cup_ordinal": user_features.user_bust_cup_ordinal,
    "user_body_type": user_features.user_body_type,
    "user_age": user_features.user_age,
    "item_category": item_features.item_category,
    "item_nominal_size": item_features.item_nominal_size,
    "user_prior_review_count": interaction_features.build_history_table,
    "user_prior_mismatch_rate": interaction_features.build_history_table,
    "user_prior_small_rate": interaction_features.build_history_table,
    "item_prior_mismatch_rate": interaction_features.build_history_table,
    "context_rented_for": context_features.context_rented_for,
}

# The four historical aggregates share a single implementation function
# (`build_history_table` computes all of them together, since they are all
# expanding statistics over the same chronologically-ordered frame), so the
# registry maps each declared name to it for the declaration/implementation
# parity check even though only one call actually happens in the pipeline.


def validate_registry(specs: list[FeatureSpec]) -> None:
    """Assert the declaration and the implementation agree in both directions.

    Args:
        specs: Parsed feature specifications.

    Raises:
        ValueError: If a declared feature has no implementation, or an
            implemented transform is never declared.
    """
    declared = {spec.transform for spec in specs}
    implemented = set(TRANSFORM_REGISTRY)

    undeclared = implemented - declared
    unimplemented = declared - implemented

    problems: list[str] = []
    if unimplemented:
        problems.append(f"declared but not implemented: {sorted(unimplemented)}")
    if undeclared:
        problems.append(f"implemented but not declared: {sorted(undeclared)}")

    if problems:
        raise ValueError("Feature registry mismatch -- " + "; ".join(problems))
