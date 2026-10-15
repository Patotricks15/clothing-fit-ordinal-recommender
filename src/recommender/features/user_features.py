"""Static per-user attribute transforms.

Every function here reads a value straight off the row being featurized
(body measurements, body type, age) -- the same way a sizing form on a
checkout page would. None of these aggregate interaction history, so none of
them need -- or accept -- a point-in-time cut-off, and none of them require
"fitting" on a training partition: a brand-new user supplies these values the
moment they fill in their profile, train or test alike.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

BUST_PATTERN = re.compile(r"(?P<band>\d{2})(?P<cup>[a-z]+)", re.IGNORECASE)
CUP_ORDER = ["aa", "a", "b", "c", "d", "dd", "ddd", "e", "f", "g"]


def user_height_cm(frame: pd.DataFrame) -> pd.Series:
    """Static height in centimetres, row-aligned."""
    series = frame["height_cm"].astype(np.float32).reset_index(drop=True)
    series.name = "user_height_cm"
    return series


def user_weight_kg(frame: pd.DataFrame) -> pd.Series:
    """Static weight in kilograms, row-aligned."""
    series = frame["weight_kg"].astype(np.float32).reset_index(drop=True)
    series.name = "user_weight_kg"
    return series


def user_bmi(frame: pd.DataFrame) -> pd.Series:
    """Body-mass index computed purely from the two static measurements."""
    height_m = frame["height_cm"].astype(np.float64) / 100.0
    bmi = (frame["weight_kg"].astype(np.float64) / (height_m**2)).astype(np.float32)
    bmi = bmi.reset_index(drop=True)
    bmi.name = "user_bmi"
    return bmi


def _parse_bust(bust_size: str) -> tuple[float, str]:
    match = BUST_PATTERN.match(str(bust_size).strip())
    if not match:
        return np.nan, ""
    return float(match.group("band")), match.group("cup").lower()


def user_bust_band(frame: pd.DataFrame) -> pd.Series:
    """Numeric chest band parsed from the static ``bust_size`` string."""
    bands = frame["bust_size"].map(lambda v: _parse_bust(v)[0]).astype(np.float32)
    bands = bands.reset_index(drop=True)
    bands.name = "user_bust_band"
    return bands


def user_bust_cup_ordinal(frame: pd.DataFrame) -> pd.Series:
    """Ordinal-encoded cup letter parsed from the static ``bust_size`` string."""

    def _cup_rank(value: str) -> float:
        cup = _parse_bust(value)[1]
        return float(CUP_ORDER.index(cup)) if cup in CUP_ORDER else -1.0

    ranks = frame["bust_size"].map(_cup_rank).astype(np.float32).reset_index(drop=True)
    ranks.name = "user_bust_cup_ordinal"
    return ranks


def user_body_type(frame: pd.DataFrame) -> pd.Series:
    """Static self-reported body-type category."""
    series = frame["body_type"].astype(object).reset_index(drop=True)
    series.name = "user_body_type"
    return series


def user_age(frame: pd.DataFrame) -> pd.Series:
    """Static demographic attribute."""
    series = frame["age"].astype(np.float32).reset_index(drop=True)
    series.name = "user_age"
    return series
