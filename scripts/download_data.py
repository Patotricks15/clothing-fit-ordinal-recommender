#!/usr/bin/env python3
"""Fetch the Clothing Fit Dataset for Size Recommendation, or generate a
synthetic fixture with the same schema.

The synthetic generator is the default so the repository is runnable from a
clean checkout with no Kaggle account. It reproduces clustered user body
types, clustered item categories/sizing, and a skewed ``fit`` distribution
(most garments "fit"; "small"/"large" are the minority classes), plus a true
``review_date`` column so point-in-time aggregate features exercise real
chronological ordering.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KAGGLE_DATASET = "rmisra/clothing-fit-dataset-for-size-recommendation"

CATEGORIES = ["dress", "top", "bottoms", "outerwear", "jumpsuit", "gown", "sweater"]
BODY_TYPES = ["hourglass", "pear", "athletic", "petite", "apple", "full bust", "straight & narrow"]
RENTED_FOR = ["wedding", "party", "work", "everyday", "formal affair", "vacation"]
BUST_BANDS = ["32", "34", "36", "38", "40"]
BUST_CUPS = ["a", "b", "c", "d", "dd"]


def generate_synthetic(
    n_users: int = 1_200,
    n_items: int = 450,
    seed: int = 20260220,
) -> pd.DataFrame:
    """Build a fixture with clustered users/items and a skewed fit distribution.

    Users are assigned to latent "size clusters" (a combination of a true
    underlying measurement scale and a body-type label); items are assigned to
    "fit-sensitivity clusters" controlling how harshly a size mismatch is
    judged. The resulting ``fit`` label is skewed towards "fit" (as in the
    real dataset, most purchases work out), with "small"/"large" probability
    rising smoothly as the user's true size diverges from the item's
    nominal size.

    Args:
        n_users: Number of synthetic users.
        n_items: Number of synthetic garments.
        seed: Seed controlling the whole generation.

    Returns:
        A tidy frame matching :data:`recommender.data.loader.REQUIRED_COLUMNS`.
    """
    rng = np.random.default_rng(seed)

    # --- Users: a true latent size plus static measurements correlated with it.
    user_true_size = rng.normal(40, 6, size=n_users).clip(28, 56)
    height_cm = (user_true_size * 1.5 + rng.normal(120, 5, size=n_users)).clip(145, 195)
    weight_kg = (user_true_size * 1.9 + rng.normal(5, 6, size=n_users)).clip(40, 130)
    body_type = rng.choice(BODY_TYPES, size=n_users)
    bust_size = np.array(
        [f"{rng.choice(BUST_BANDS)}{rng.choice(BUST_CUPS)}" for _ in range(n_users)]
    )
    age = rng.integers(18, 65, size=n_users)
    user_ids = np.arange(1, n_users + 1)

    # --- Items: nominal size plus a fit-sensitivity (how unforgiving the cut is).
    item_nominal_size = rng.normal(40, 6, size=n_items).clip(28, 56)
    item_sensitivity = rng.uniform(0.6, 1.6, size=n_items)
    item_category = rng.choice(CATEGORIES, size=n_items)
    item_ids = np.arange(1, n_items + 1)

    records: list[dict[str, object]] = []
    base_date = datetime(2023, 1, 1)

    for u in range(n_users):
        n_orders = int(rng.integers(3, 18))

        # Users mostly self-select garments close to their own true size (as
        # shoppers do in practice), with enough spread that mismatches -- and
        # therefore "small"/"large" outcomes -- still occur regularly.
        closeness = np.exp(-0.5 * ((item_nominal_size - user_true_size[u]) / 3.2) ** 2)
        weights = closeness + 0.08
        weights = weights / weights.sum()
        chosen_items = rng.choice(n_items, size=n_orders, replace=False, p=weights)
        # Orders increase in time within a user.
        offsets = np.sort(rng.integers(0, 700, size=n_orders))

        for position, item_idx in enumerate(chosen_items):
            nominal = item_nominal_size[item_idx]
            sensitivity = item_sensitivity[item_idx]
            true_size = user_true_size[u]

            # Garment "size" purchased is the nominal size rounded to a
            # standard grid, with light noise reflecting imperfect self-selection.
            purchased_size = int(round((nominal + rng.normal(0, 1.5)) / 2) * 2)
            purchased_size = int(np.clip(purchased_size, 28, 56))

            gap = (true_size - purchased_size) * sensitivity
            # Skewed towards "fit": only a sizeable gap pushes small/large.
            p_small = 1.0 / (1.0 + np.exp(-(gap - 4.5) / 2.0))
            p_large = 1.0 / (1.0 + np.exp(-(-gap - 4.5) / 2.0))
            p_fit = max(1e-3, 1.0 - p_small - p_large)
            probs = np.array([p_small, p_fit, p_large])
            probs = probs / probs.sum()
            fit = rng.choice(["small", "fit", "large"], p=probs)

            review_date = base_date + timedelta(days=int(offsets[position]))

            records.append(
                {
                    "user_id": int(user_ids[u]),
                    "item_id": int(item_ids[item_idx]),
                    "category": item_category[item_idx],
                    "size": purchased_size,
                    "fit": fit,
                    "height_cm": round(float(height_cm[u]), 1),
                    "weight_kg": round(float(weight_kg[u]), 1),
                    "bust_size": bust_size[u],
                    "body_type": body_type[u],
                    "age": int(age[u]),
                    "rented_for": rng.choice(RENTED_FOR),
                    "review_date": review_date.strftime("%Y-%m-%d"),
                }
            )

    return pd.DataFrame.from_records(records)


def _normalize_real_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the Kaggle ModCloth/RentTheRunway schema onto our column contract."""
    frame = raw.rename(
        columns={
            "user_id": "user_id",
            "item_id": "item_id",
            "weight": "weight_kg",
            "height": "height_cm",
            "bust size": "bust_size",
            "body type": "body_type",
            "rented for": "rented_for",
            "review_date": "review_date",
        }
    )
    frame["category"] = frame.get("category", "unknown")
    return frame


def download_from_kaggle(target: Path) -> None:
    """Download the real dataset using credentials from the environment."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment.\n"
            "Credentials are never read from or written to this repository."
        )

    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET, "-p", str(target), "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("The 'kaggle' package is not installed: pip install kaggle") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Kaggle download failed with exit code {exc.returncode}") from exc

    print(
        f"Downloaded {KAGGLE_DATASET} into {target}.\n"
        "Run 'python scripts/normalize_kaggle_export.py' (or adapt "
        "_normalize_real_dataset in this script) to produce fit_records.csv."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true",
        help="Generate the offline fixture (default).",
    )
    source.add_argument(
        "--kaggle", action="store_true",
        help="Download the real dataset using KAGGLE_USERNAME/KAGGLE_KEY.",
    )
    parser.add_argument("--users", type=int, default=1_200)
    parser.add_argument("--items", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260220)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        download_from_kaggle(DATA_DIR)
        return 0

    frame = generate_synthetic(args.users, args.items, args.seed)
    frame.to_csv(DATA_DIR / "fit_records.csv", index=False)

    fit_counts = frame["fit"].value_counts(normalize=True).round(3).to_dict()
    print(
        f"Synthetic fixture written to {DATA_DIR / 'fit_records.csv'}\n"
        f"  users={args.users} items={args.items} records={len(frame)}\n"
        f"  fit distribution: {fit_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
