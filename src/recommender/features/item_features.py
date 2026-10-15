"""Static per-item attribute transforms.

``item_category`` is read straight off the row, like a product listing field.
``item_nominal_size`` is a catalogue-level summary (the modal purchased size
across *every* review of the item, train or not) -- it is fit on the whole
catalogue rather than per row because, unlike the category label, no single
review carries the garment's overall sizing behaviour; that only emerges by
aggregating across reviews of the same item. It carries no point-in-time
cut-off because it describes the garment's cut, analogous to a
manufacturer's size chart, not any individual transaction's outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_category(frame: pd.DataFrame) -> pd.Series:
    """Garment category, read directly from each row."""
    series = frame["category"].fillna("unknown").astype(object).reset_index(drop=True)
    series.name = "item_category"
    return series


def fit_item_nominal_size_table(catalogue: pd.DataFrame, n_items: int) -> pd.Series:
    """Fit the catalogue-wide modal size lookup, indexed by ``item_index``.

    Args:
        catalogue: Every known review row (used only for its ``item_index``/
            ``size`` columns; see the module docstring for why this needs
            the whole catalogue rather than a single row).
        n_items: Total number of items, for a dense, cold-start-safe index.

    Returns:
        Per-item modal size, with unseen items back-filled by the global mean.
    """
    modal = catalogue.groupby("item_index")["size"].median()
    values = pd.Series(np.nan, index=pd.RangeIndex(n_items), dtype=np.float64)
    values.loc[modal.index] = modal.to_numpy()

    global_mean = float(np.nanmean(values.to_numpy())) if values.notna().any() else 40.0
    values = values.fillna(global_mean).astype(np.float32)
    values.name = "item_nominal_size"
    return values


def item_nominal_size(frame: pd.DataFrame, lookup_table: pd.Series) -> pd.Series:
    """Look up the fitted modal size for each row's ``item_index``.

    Unseen item indices (cold-start items beyond the fitted table's range)
    fall back to the table's own global mean rather than propagating NaN.
    """
    fallback = float(lookup_table.mean())
    values = lookup_table.reindex(frame["item_index"].to_numpy()).fillna(fallback)
    series = pd.Series(values.to_numpy(dtype=np.float32), index=pd.RangeIndex(len(frame)))
    series.name = "item_nominal_size"
    return series
