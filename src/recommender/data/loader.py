"""Loading and validation for the Clothing Fit Dataset for Size Recommendation.

The raw Kaggle dataset (``rmisra/clothing-fit-dataset-for-size-recommendation``)
ships one JSON-lines file per marketplace (ModCloth, RentTheRunway) with
fields that include user body measurements (``height``, ``weight``,
``bust size``, ``body type``), an item identifier and ``category``, a garment
``size``, and the ordinal outcome ``fit`` in ``{"small", "fit", "large"}``.

This module normalizes that schema (real or synthetic) into a single tidy
frame with a stable column contract, and assigns a contiguous
``review_order`` per user derived from ``review_date`` so point-in-time
aggregate features (Section "Feature families" in the README) can be computed
with a strict ``<`` comparison against a cut-off position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ordinal encoding: small < fit < large. This ordering is the one contract the
# rest of the codebase (features, model, conformal, evaluation) must agree on.
FIT_LEVELS: tuple[str, ...] = ("small", "fit", "large")
FIT_LEVEL_TO_CODE: dict[str, int] = {level: idx for idx, level in enumerate(FIT_LEVELS)}

REQUIRED_COLUMNS = {
    "user_id",
    "item_id",
    "category",
    "size",
    "fit",
    "height_cm",
    "weight_kg",
    "bust_size",
    "body_type",
    "age",
    "review_date",
}


@dataclass(frozen=True)
class FitData:
    """Indexed, validated clothing-fit interactions.

    Attributes:
        frame: Tidy records with ``user_index``/``item_index`` and
            ``review_order`` (position within each user's history, ordered by
            ``review_date``) plus ``fit_code`` (0=small, 1=fit, 2=large).
        user_ids: Original user ids, positionally aligned to ``user_index``.
        item_ids: Original item ids, positionally aligned to ``item_index``.
    """

    frame: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_fit_records(path: str | Path) -> pd.DataFrame:
    """Read the normalized clothing-fit CSV and validate its contract.

    Args:
        path: Path to ``fit_records.csv`` (real or synthetic).

    Returns:
        The raw frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or ``fit`` has unknown
            levels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )

    frame = pd.read_csv(path, parse_dates=["review_date"])
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    unknown_fit = set(frame["fit"].unique()) - set(FIT_LEVELS)
    if unknown_fit:
        raise ValueError(f"{path} contains unknown fit levels: {sorted(unknown_fit)}")

    return frame


def build_indexed_dataset(records: pd.DataFrame) -> FitData:
    """Assign contiguous indices and a point-in-time review order.

    Args:
        records: Raw frame from :func:`load_fit_records`.

    Returns:
        An indexed :class:`FitData` sorted by user then ``review_date``.
    """
    frame = records.drop_duplicates(subset=["user_id", "item_id", "review_date"]).copy()
    frame = frame.sort_values(["user_id", "review_date"], kind="mergesort").reset_index(drop=True)

    user_ids = np.sort(frame["user_id"].unique())
    item_ids = np.sort(frame["item_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    frame["user_index"] = frame["user_id"].map(user_map).astype(np.int32)
    frame["item_index"] = frame["item_id"].map(item_map).astype(np.int32)
    frame["review_order"] = frame.groupby("user_index").cumcount()
    frame["fit_code"] = frame["fit"].map(FIT_LEVEL_TO_CODE).astype(np.int8)

    return FitData(frame=frame, user_ids=user_ids, item_ids=item_ids)
