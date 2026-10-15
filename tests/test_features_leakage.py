"""Leakage tests.

These are the most important tests in the repository. A recommender that
leaks future information reports excellent offline metrics and then fails in
production, so each leakage class gets an explicit, failing-by-default
check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import build_indexed_dataset
from recommender.features import interaction_features
from recommender.features.pipeline import assign_global_order


def test_future_review_does_not_change_past_user_mismatch_rate(dataset) -> None:
    """Injecting a future review for an existing user must not alter earlier rows.

    This is the direct operationalization of the point-in-time rule: the
    expanding aggregate at position ``review_order = k`` must be identical
    whether or not a review at a later position exists.
    """
    ordered = assign_global_order(dataset.frame)
    before = interaction_features.build_history_table(ordered)

    victim = int(ordered["user_index"].iloc[0])
    victim_rows = ordered.loc[ordered["user_index"] == victim]
    max_order = int(victim_rows["review_order"].max())

    future_row = ordered.iloc[[0]].copy()
    future_row["user_index"] = victim
    future_row["review_order"] = max_order + 50
    future_row["global_order"] = int(ordered["global_order"].max()) + 1
    future_row["fit"] = "small"  # an extreme outcome that would move any mean
    polluted = pd.concat([ordered, future_row], ignore_index=True)

    after = interaction_features.build_history_table(polluted)

    before_victim = before.loc[
        (before["user_index"] == victim) & (before["review_order"] <= max_order)
    ].sort_values("review_order")
    after_victim = after.loc[
        (after["user_index"] == victim) & (after["review_order"] <= max_order)
    ].sort_values("review_order")

    pd.testing.assert_series_equal(
        before_victim["user_prior_mismatch_rate"].reset_index(drop=True),
        after_victim["user_prior_mismatch_rate"].reset_index(drop=True),
        check_exact=True,
    )


def test_future_review_does_not_change_past_item_mismatch_rate(dataset) -> None:
    ordered = assign_global_order(dataset.frame)
    before = interaction_features.build_history_table(ordered)

    victim_item = int(ordered["item_index"].iloc[0])
    max_global = int(ordered["global_order"].max())

    future_row = ordered.iloc[[0]].copy()
    future_row["item_index"] = victim_item
    future_row["global_order"] = max_global + 100
    future_row["review_order"] = int(ordered["review_order"].max()) + 1
    future_row["fit"] = "large"
    polluted = pd.concat([ordered, future_row], ignore_index=True)

    after = interaction_features.build_history_table(polluted)

    before_victim = before.loc[
        (before["item_index"] == victim_item) & (before["global_order"] <= max_global)
    ].sort_values("global_order")
    after_victim = after.loc[
        (after["item_index"] == victim_item) & (after["global_order"] <= max_global)
    ].sort_values("global_order")

    pd.testing.assert_series_equal(
        before_victim["item_prior_mismatch_rate"].reset_index(drop=True),
        after_victim["item_prior_mismatch_rate"].reset_index(drop=True),
        check_exact=True,
    )


def test_cutoff_boundary_is_strictly_exclusive(dataset) -> None:
    """A review exactly at the cut-off belongs to the future, not the past."""
    ordered = assign_global_order(dataset.frame)
    history = interaction_features.build_history_table(ordered)

    user = int(ordered["user_index"].iloc[0])
    user_rows = history.loc[history["user_index"] == user].sort_values("review_order")

    # The second row's prior count must equal exactly 1 (only the first row
    # counts), never 2 (which would mean the row counted itself).
    assert user_rows.iloc[0]["user_prior_review_count"] == 0
    if len(user_rows) > 1:
        assert user_rows.iloc[1]["user_prior_review_count"] == 1


def test_first_review_has_no_prior_history(dataset) -> None:
    ordered = assign_global_order(dataset.frame)
    history = interaction_features.build_history_table(ordered)
    first_reviews = history.loc[history["review_order"] == 0]
    assert (first_reviews["user_prior_review_count"] == 0).all()


def test_historical_feature_differs_from_naive_full_history_average(dataset) -> None:
    """The point-in-time rate must differ from a naive whole-history average.

    If they were identical, every row would see its own future reviews
    folded into its own feature, which is exactly the leak this project is
    built to avoid.
    """
    ordered = assign_global_order(dataset.frame)
    history = interaction_features.build_history_table(ordered)

    multi_review_users = history["user_index"].value_counts()
    multi_review_users = multi_review_users[multi_review_users > 3].index
    assert len(multi_review_users) > 0, "fixture must contain users with multiple reviews"

    naive = history.groupby("user_index")["user_prior_mismatch_rate"].transform("mean")
    pit = history["user_prior_mismatch_rate"]

    sample = history["user_index"].isin(multi_review_users)
    assert not np.allclose(
        naive.loc[sample].to_numpy(), pit.loc[sample].to_numpy(), equal_nan=True
    )


def test_grouped_split_rejects_a_user_appearing_in_two_partitions(dataset) -> None:
    from recommender.data.splitting import GroupSplit

    frame = dataset.frame
    half = frame.iloc[: len(frame) // 2]
    other_half = frame.iloc[len(frame) // 2 :]

    with pytest.raises(ValueError, match="leaked"):
        GroupSplit(
            train=half,
            validation=other_half,
            test=frame.iloc[:5],
            n_users=dataset.n_users,
            n_items=dataset.n_items,
        )


def test_no_duplicate_reviews_before_splitting(raw_records) -> None:
    data = build_indexed_dataset(raw_records)
    frame = data.frame
    assert not frame.duplicated(subset=["user_index", "item_index", "review_date"]).any()
