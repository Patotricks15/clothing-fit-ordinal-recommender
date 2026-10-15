"""Tests for data loading and the grouped user split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import FIT_LEVELS, grouped_user_split


def test_fit_levels_are_ordered() -> None:
    assert FIT_LEVELS == ("small", "fit", "large")


def test_build_indexed_dataset_assigns_contiguous_indices(dataset) -> None:
    assert dataset.frame["user_index"].max() == dataset.n_users - 1
    assert dataset.frame["item_index"].max() == dataset.n_items - 1
    assert dataset.frame["fit_code"].isin([0, 1, 2]).all()


def test_review_order_is_strictly_increasing_per_user(dataset) -> None:
    """The frame is already sorted by (user, review_date); review_order must
    follow that existing row order exactly -- re-sorting by review_date here
    would be ambiguous whenever two reviews share the same date.
    """
    frame = dataset.frame
    for _, group in frame.groupby("user_index", sort=False):
        orders = group["review_order"].to_numpy()
        assert (orders == np.arange(len(orders))).all()
        assert len(set(orders)) == len(orders)


def test_grouped_split_has_no_overlapping_users(split) -> None:
    train_users = set(split.train["user_index"].unique())
    val_users = set(split.validation["user_index"].unique())
    test_users = set(split.test["user_index"].unique())

    assert not (train_users & val_users)
    assert not (train_users & test_users)
    assert not (val_users & test_users)


def test_grouped_split_covers_every_user(split, dataset) -> None:
    all_users = set(split.train["user_index"].unique())
    all_users |= set(split.validation["user_index"].unique())
    all_users |= set(split.test["user_index"].unique())
    assert all_users == set(range(dataset.n_users))


def test_grouped_split_rejects_degenerate_fractions(dataset) -> None:
    with pytest.raises(ValueError):
        grouped_user_split(dataset, validation_frac=0.5, test_frac=0.6)


def test_a_single_users_measurements_never_split_across_partitions(split) -> None:
    """The leakage mode unique to this dataset: static per-user attributes."""
    for partition in (split.train, split.validation, split.test):
        cols = ["height_cm", "weight_kg", "body_type"]
        per_user = partition.groupby("user_index")[cols].nunique()
        assert (per_user <= 1).all().all(), "a user's static attributes vary within one partition"


def test_no_duplicate_review_rows(dataset) -> None:
    frame = dataset.frame
    assert not frame.duplicated(subset=["user_index", "item_index", "review_date"]).any()


def test_raw_records_round_trip(raw_records: pd.DataFrame) -> None:
    assert {"user_id", "item_id", "fit", "height_cm", "weight_kg"}.issubset(raw_records.columns)
    assert raw_records["fit"].isin(FIT_LEVELS).all()
