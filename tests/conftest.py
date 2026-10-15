"""Shared pytest fixtures built from the synthetic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from recommender.data import build_indexed_dataset, grouped_user_split, load_fit_records

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _ensure_dataset() -> None:
    """Generate the synthetic fixture on demand so tests are self-contained."""
    if (DATA_DIR / "fit_records.csv").exists():
        return

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from download_data import generate_synthetic

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic(n_users=260, n_items=120, seed=11)
    frame.to_csv(DATA_DIR / "fit_records.csv", index=False)


@pytest.fixture(scope="session")
def raw_records() -> pd.DataFrame:
    _ensure_dataset()
    return load_fit_records(DATA_DIR / "fit_records.csv")


@pytest.fixture(scope="session")
def dataset(raw_records: pd.DataFrame):
    return build_indexed_dataset(raw_records)


@pytest.fixture(scope="session")
def split(dataset):
    return grouped_user_split(dataset, random_state=0)


@pytest.fixture(scope="session")
def feature_store(dataset):
    from recommender.features import build_features

    return build_features(dataset)
