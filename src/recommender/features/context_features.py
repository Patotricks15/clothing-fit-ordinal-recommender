"""Per-row context transforms.

Context features describe the current transaction (e.g. the occasion the
garment is rented for), not a history to aggregate. They are supplied by the
shopper at request time -- like a search filter -- so they carry no
point-in-time cut-off: the value exists at the moment of the row itself.
"""

from __future__ import annotations

import pandas as pd


def context_rented_for(frame: pd.DataFrame) -> pd.Series:
    """Pass through the stated occasion for each row, row-index aligned."""
    series = frame["rented_for"].fillna("unknown").astype(object).reset_index(drop=True)
    series.name = "context_rented_for"
    return series
