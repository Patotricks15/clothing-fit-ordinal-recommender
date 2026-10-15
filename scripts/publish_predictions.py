#!/usr/bin/env python3
"""Train, calibrate and publish precomputed predictions to DynamoDB/Floci.

Run the local Floci emulator first (``docker compose up -d``), then:

    python scripts/publish_predictions.py --endpoint-url http://localhost:4566
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender.cli import _design_matrices, _load_split  # noqa: E402
from recommender.models import OrdinalLogisticRegression, SplitConformalClassifier, Standardizer  # noqa: E402
from recommender.serving import build_prediction_payloads  # noqa: E402

ORDINAL_L2 = 2.0
CONFORMAL_ALPHA = 0.1


def _to_dynamo_safe(value):  # noqa: ANN001, ANN202
    """Recursively convert Python floats to Decimal; DynamoDB rejects float."""
    from decimal import Decimal

    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dynamo_safe(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "fit_records.csv")
    parser.add_argument("--table", default="clothing-fit-dev-predictions")
    parser.add_argument("--endpoint-url", default="http://localhost:4566")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=200, help="Number of test rows to publish.")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run scripts/download_data.py --synthetic")

    dataset, split, feature_store = _load_split(args.data, args.seed)
    (train_X, train_y), (val_X, val_y), _ = _design_matrices(feature_store, split)

    scaler = Standardizer().fit(train_X)
    model = OrdinalLogisticRegression(l2=ORDINAL_L2).fit(scaler.transform(train_X), train_y)
    conformal = SplitConformalClassifier(alpha=CONFORMAL_ALPHA).calibrate(
        model.predict_proba(scaler.transform(val_X)), val_y
    )

    rows = split.test.iloc[: args.limit]
    payloads = build_prediction_payloads(
        model, scaler, conformal, feature_store, rows, dataset.user_ids, dataset.item_ids
    )

    import boto3

    resource = boto3.resource("dynamodb", endpoint_url=args.endpoint_url, region_name="us-east-1")
    table = resource.Table(args.table)
    with table.batch_writer() as batch:
        for payload in payloads:
            batch.put_item(Item=_to_dynamo_safe(payload))

    print(
        f"published {len(payloads)} predictions -> "
        f"table={args.table} endpoint={args.endpoint_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
