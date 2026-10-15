"""Grouped-by-user train/validation/test splitting.

Unlike a collaborative-filtering leave-last-one-out split, every record here
carries static user body measurements (height, weight, bust size, body
type). If the *same user* appeared in both train and test, the model could
memorize that user's measurements during training and then trivially match
them at evaluation time -- a leakage mode that is specific to supervised
tabular problems with repeated subjects, not just to time-ordered
aggregates. The split below therefore partitions at the **user** level: a
user's entire history goes to exactly one of train/validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import FitData


@dataclass(frozen=True)
class GroupSplit:
    """A user-disjoint partition of the clothing-fit interactions.

    Attributes:
        train: Rows belonging to the training users.
        validation: Rows belonging to the validation users.
        test: Rows belonging to the test users.
        n_users: Total number of users in the dataset.
        n_items: Total number of items in the dataset.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int

    def __post_init__(self) -> None:
        train_users = set(self.train["user_index"].unique())
        val_users = set(self.validation["user_index"].unique())
        test_users = set(self.test["user_index"].unique())
        overlap = (train_users & val_users) | (train_users & test_users) | (val_users & test_users)
        if overlap:
            raise ValueError(
                f"Grouped split leaked {len(overlap)} user(s) across partitions: "
                f"{sorted(overlap)[:5]}..."
            )


def grouped_user_split(
    data: FitData,
    validation_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 0,
) -> GroupSplit:
    """Partition users (not rows) into train/validation/test.

    Args:
        data: Indexed interactions.
        validation_frac: Fraction of *users* assigned to validation.
        test_frac: Fraction of *users* assigned to test.
        random_state: Seed for the user-level shuffle.

    Returns:
        A :class:`GroupSplit` whose partitions share no user.

    Raises:
        ValueError: If the requested fractions leave no users for training.
    """
    if not 0 < validation_frac < 1 or not 0 < test_frac < 1:
        raise ValueError("validation_frac and test_frac must be in (0, 1).")
    if validation_frac + test_frac >= 1:
        raise ValueError("validation_frac + test_frac must be < 1.")

    rng = np.random.default_rng(random_state)
    users = np.arange(data.n_users)
    rng.shuffle(users)

    n_val = max(1, int(round(data.n_users * validation_frac)))
    n_test = max(1, int(round(data.n_users * test_frac)))
    if n_val + n_test >= data.n_users:
        raise ValueError("validation_frac + test_frac leaves no users for training.")

    val_users = set(users[:n_val].tolist())
    test_users = set(users[n_val : n_val + n_test].tolist())
    train_users = set(users[n_val + n_test :].tolist())

    frame = data.frame
    train = frame.loc[frame["user_index"].isin(train_users)].reset_index(drop=True)
    validation = frame.loc[frame["user_index"].isin(val_users)].reset_index(drop=True)
    test = frame.loc[frame["user_index"].isin(test_users)].reset_index(drop=True)

    return GroupSplit(
        train=train, validation=validation, test=test, n_users=data.n_users, n_items=data.n_items
    )
