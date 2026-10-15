"""Point-in-time historical aggregate transforms.

Each of these is a per-row *expanding* statistic: the value attached to a
given review depends only on reviews that happened strictly earlier,
according to an explicit chronological order column, and never on the
review's own outcome or on anything that happened later. Because the cutoff
is intrinsic to each row's position in history, these features need no
separate "fit" step and no train/validation/test distinction to stay safe --
the same deterministic computation is correct whether the row being
featurized sits in the train, validation or test partition. What must never
happen is computing the statistic over an *unordered* group (which would let
a later review influence an earlier one); the explicit ``sort_values`` below
is what prevents that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISMATCH_LEVELS = ("small", "large")

HISTORY_COLUMNS = [
    "user_prior_review_count",
    "user_prior_mismatch_rate",
    "user_prior_small_rate",
    "item_prior_mismatch_rate",
]


def _expanding_prior_mean(
    frame: pd.DataFrame, group_col: str, order_col: str, value_col: str
) -> np.ndarray:
    """Mean of ``value_col`` over strictly-prior rows within each group.

    Args:
        frame: The frame to compute over (re-sorted defensively here; the
            result is reindexed back to ``frame``'s original row order).
        group_col: Column identifying the entity (user or item).
        order_col: Strictly increasing chronological position within the
            group.
        value_col: Numeric (0/1) column to average.

    Returns:
        An array aligned to ``frame``'s original row order. ``NaN`` where no
        prior row exists for that group (the first observation).
    """
    ordered = frame.sort_values([group_col, order_col], kind="mergesort")
    grouped = ordered.groupby(group_col)[value_col]
    cumulative_sum = grouped.cumsum() - ordered[value_col]
    prior_count = grouped.cumcount()
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(prior_count > 0, cumulative_sum / prior_count, np.nan)
    result = pd.Series(prior_mean, index=ordered.index)
    return result.reindex(frame.index).to_numpy()


def build_history_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the four point-in-time historical aggregates to every row.

    Args:
        frame: The full historical record (every known review, across every
            partition), carrying ``user_index``, ``item_index``,
            ``review_order`` (position within the user's own history) and
            ``global_order`` (position within the whole dataset).

    Returns:
        ``frame`` with the four columns in :data:`HISTORY_COLUMNS` appended.
        Rows with no prior history get ``NaN``; the caller (the feature
        pipeline) is responsible for applying the declared fill policy.
    """
    frame = frame.copy()
    frame["_is_mismatch"] = frame["fit"].isin(MISMATCH_LEVELS).astype(np.float64)
    frame["_is_small"] = (frame["fit"] == "small").astype(np.float64)

    ordered_by_user = frame.sort_values(["user_index", "review_order"], kind="mergesort")
    prior_count = ordered_by_user.groupby("user_index").cumcount()
    frame["user_prior_review_count"] = (
        pd.Series(prior_count.to_numpy(), index=ordered_by_user.index)
        .reindex(frame.index)
        .astype(np.float32)
    )

    frame["user_prior_mismatch_rate"] = _expanding_prior_mean(
        frame, "user_index", "review_order", "_is_mismatch"
    ).astype(np.float32)
    frame["user_prior_small_rate"] = _expanding_prior_mean(
        frame, "user_index", "review_order", "_is_small"
    ).astype(np.float32)
    frame["item_prior_mismatch_rate"] = _expanding_prior_mean(
        frame, "item_index", "global_order", "_is_mismatch"
    ).astype(np.float32)

    return frame.drop(columns=["_is_mismatch", "_is_small"])
