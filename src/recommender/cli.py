"""Command line interface: train, evaluate and predict clothing fit.

Hyperparameters below were selected on the *validation* partition; the test
partition is touched only by ``evaluate``, and only once, which is what
keeps the reported numbers honest.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data import FIT_LEVELS, build_indexed_dataset, grouped_user_split, load_fit_records
from recommender.evaluation import evaluate_classification, evaluate_conformal
from recommender.features import build_features
from recommender.models import (
    MajorityClassClassifier,
    MultinomialLogisticRegression,
    OrdinalLogisticRegression,
    SplitConformalClassifier,
    Standardizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_DATA = PROJECT_ROOT / "data" / "fit_records.csv"

# Selected on the validation partition.
ORDINAL_L2 = 2.0
MLR_L2 = 2.0
CONFORMAL_ALPHA = 0.1


def _load_split(data_path: Path, seed: int):
    records = load_fit_records(data_path)
    dataset = build_indexed_dataset(records)
    split = grouped_user_split(dataset, random_state=seed)
    feature_store = build_features(dataset)
    return dataset, split, feature_store


def _design_matrices(feature_store, split):
    train_X = feature_store.encode(feature_store.transform(split.train))
    val_X = feature_store.encode(feature_store.transform(split.validation))
    test_X = feature_store.encode(feature_store.transform(split.test))
    train_y = split.train["fit_code"].to_numpy()
    val_y = split.validation["fit_code"].to_numpy()
    test_y = split.test["fit_code"].to_numpy()
    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


def _build_models(seed: int) -> dict:  # noqa: ARG001
    return {
        "Majority": MajorityClassClassifier(n_classes=len(FIT_LEVELS)),
        "MultinomialLR": MultinomialLogisticRegression(n_classes=len(FIT_LEVELS), l2=MLR_L2),
        "OrdinalLR": OrdinalLogisticRegression(n_classes=len(FIT_LEVELS), l2=ORDINAL_L2),
    }


def cmd_train(args: argparse.Namespace) -> int:
    _, split, feature_store = _load_split(args.data, args.seed)
    (train_X, train_y), _, _ = _design_matrices(feature_store, split)

    scaler = Standardizer().fit(train_X)
    model = OrdinalLogisticRegression(n_classes=len(FIT_LEVELS), l2=ORDINAL_L2)
    model.fit(scaler.transform(train_X), train_y)

    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "ordinal_model.pkl").open("wb") as handle:
        pickle.dump(
            {"model": model, "scaler": scaler, "feature_store": feature_store}, handle
        )

    print(f"users={split.n_users}  items={split.n_items}  train_rows={len(split.train)}")
    print(f"OrdinalLR converged={model.converged_}  thresholds={model.thresholds_}")
    print(f"saved -> {ARTIFACTS / 'ordinal_model.pkl'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _, split, feature_store = _load_split(args.data, args.seed)
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = _design_matrices(feature_store, split)

    scaler = Standardizer().fit(train_X)
    train_Xs, val_Xs, test_Xs = (
        scaler.transform(train_X),
        scaler.transform(val_X),
        scaler.transform(test_X),
    )

    eval_X, eval_y = (val_Xs, val_y) if args.partition == "validation" else (test_Xs, test_y)

    rows, conformal_rows = [], []
    for name, model in _build_models(args.seed).items():
        model.fit(train_Xs, train_y)
        preds = model.predict(eval_X)
        report = evaluate_classification(eval_y, preds, n_classes=len(FIT_LEVELS), model_name=name)
        rows.append(report.as_row())
        print(
            f"{name:16} acc={report.accuracy:.4f}  mae={report.mean_absolute_ordinal_error:.4f}  "
            f"within1={report.accuracy_within_one:.4f}  f1={report.macro_f1:.4f}"
        )

        if hasattr(model, "predict_proba"):
            conformal = SplitConformalClassifier(alpha=CONFORMAL_ALPHA)
            conformal.calibrate(model.predict_proba(val_Xs), val_y)
            conformal_report = evaluate_conformal(
                conformal, model.predict_proba(eval_X), eval_y, model_name=name
            )
            conformal_rows.append(conformal_report.as_row())
            print(
                f"{'':16} conformal: target={conformal_report.target_coverage:.2f}  "
                f"empirical={conformal_report.empirical_coverage:.4f}  "
                f"avg_set_size={conformal_report.average_set_size:.3f}"
            )

    ARTIFACTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ARTIFACTS / "results.csv", index=False)
    pd.DataFrame(conformal_rows).to_csv(ARTIFACTS / "conformal_results.csv", index=False)
    print(f"\nsaved -> {ARTIFACTS / 'results.csv'}")
    print(f"saved -> {ARTIFACTS / 'conformal_results.csv'}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    with args.model.open("rb") as handle:
        bundle = pickle.load(handle)
    model, scaler, feature_store = bundle["model"], bundle["scaler"], bundle["feature_store"]

    request = pd.Series(
        {
            "user_index": 0,
            "item_index": 0,
            "review_order": 10**9,  # treated as "no prior history" at serving time
            "global_order": 10**9,
            "height_cm": args.height_cm,
            "weight_kg": args.weight_kg,
            "bust_size": args.bust_size,
            "body_type": args.body_type,
            "age": args.age,
            "category": args.category,
            "size": args.size,
            "rented_for": args.rented_for,
        }
    )
    features = pd.DataFrame([feature_store.row_vector(request)])
    X = scaler.transform(feature_store.encode(features))
    probs = model.predict_proba(X)[0]

    payload = {
        "predicted_fit": FIT_LEVELS[int(np.argmax(probs))],
        "probabilities": dict(zip(FIT_LEVELS, [round(float(p), 4) for p in probs], strict=True)),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clothing-fit", description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Fit the ordinal model and persist the artifact.")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="Benchmark the ordinal model against the baselines.")
    ev.add_argument("--partition", choices=["validation", "test"], default="test")
    ev.set_defaults(func=cmd_evaluate)

    pred = sub.add_parser("predict", help="Predict fit + conformal set for one request.")
    pred.add_argument("--model", type=Path, default=ARTIFACTS / "ordinal_model.pkl")
    pred.add_argument("--height-cm", type=float, required=True)
    pred.add_argument("--weight-kg", type=float, required=True)
    pred.add_argument("--bust-size", type=str, default="34b")
    pred.add_argument("--body-type", type=str, default="hourglass")
    pred.add_argument("--age", type=int, default=30)
    pred.add_argument("--category", type=str, default="dress")
    pred.add_argument("--size", type=int, default=40)
    pred.add_argument("--rented-for", type=str, default="wedding")
    pred.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run: python scripts/download_data.py --synthetic")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
