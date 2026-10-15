"""Serving tests: payload shape, JSON-safety, and handler request validation."""

from __future__ import annotations

import json

import pytest

from recommender.models import OrdinalLogisticRegression, SplitConformalClassifier, Standardizer
from recommender.serving import build_prediction_payloads, lambda_handler, parse_request
from recommender.serving.handler import _DynamoEncoder


@pytest.fixture(scope="module")
def trained_bundle(feature_store, split):
    train_X = feature_store.encode(feature_store.transform(split.train))
    val_X = feature_store.encode(feature_store.transform(split.validation))
    train_y = split.train["fit_code"].to_numpy()
    val_y = split.validation["fit_code"].to_numpy()

    scaler = Standardizer().fit(train_X)
    model = OrdinalLogisticRegression(l2=2.0).fit(scaler.transform(train_X), train_y)
    conformal = SplitConformalClassifier(alpha=0.1).calibrate(
        model.predict_proba(scaler.transform(val_X)), val_y
    )
    return model, scaler, conformal


@pytest.fixture(scope="module")
def payloads(trained_bundle, feature_store, split, dataset):
    model, scaler, conformal = trained_bundle
    return build_prediction_payloads(
        model, scaler, conformal, feature_store,
        split.test.iloc[:20], dataset.user_ids, dataset.item_ids,
    )


def test_every_requested_row_gets_a_payload(payloads, split) -> None:
    assert len(payloads) == min(20, len(split.test))


def test_payload_contains_a_valid_fit_level(payloads) -> None:
    for payload in payloads:
        assert payload["predicted_fit"] in {"small", "fit", "large"}
        assert set(payload["probabilities"]) == {"small", "fit", "large"}


def test_conformal_set_always_contains_the_argmax_class(payloads) -> None:
    """The highest-probability class can never be excluded from its own set."""
    for payload in payloads:
        assert payload["predicted_fit"] in payload["conformal_set"]


def test_probabilities_sum_to_one(payloads) -> None:
    for payload in payloads:
        assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-2)


def test_request_ids_are_unique(payloads) -> None:
    ids = [p["request_id"] for p in payloads]
    assert len(ids) == len(set(ids))


def test_payload_is_json_serialisable(payloads) -> None:
    json.dumps(payloads)


def test_build_prediction_payloads_rejects_misaligned_ids(
    trained_bundle, feature_store, split, dataset
) -> None:
    model, scaler, conformal = trained_bundle
    with pytest.raises(ValueError):
        build_prediction_payloads(
            model, scaler, conformal, feature_store,
            split.test.iloc[:5], dataset.user_ids[:1], dataset.item_ids,
        )


# --- Request validation (no AWS required) ------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        {"queryStringParameters": {}},
        {"queryStringParameters": {"request_id": ""}},
        {"queryStringParameters": None},
    ],
    ids=["missing", "empty", "null-params"],
)
def test_missing_request_id_is_rejected(event) -> None:
    with pytest.raises(ValueError, match="request_id"):
        parse_request(event)


def test_valid_request_is_parsed() -> None:
    assert parse_request({"queryStringParameters": {"request_id": "7#12#40"}}) == "7#12#40"


def test_handler_returns_400_without_touching_dynamodb() -> None:
    """A malformed request must fail fast, before any network call."""
    response = lambda_handler({"queryStringParameters": {}})
    assert response["statusCode"] == 400
    assert "request_id" in json.loads(response["body"])["error"]


def test_response_serialises_dynamodb_decimals() -> None:
    """DynamoDB returns every number as Decimal, which json.dumps rejects."""
    from decimal import Decimal

    body = {
        "request_id": "7#12#40",
        "size": Decimal("40"),
        "probabilities": {"fit": Decimal("0.5")},
    }
    decoded = json.loads(json.dumps(body, cls=_DynamoEncoder))

    assert decoded["size"] == 40
    assert isinstance(decoded["size"], int)
    assert decoded["probabilities"]["fit"] == 0.5


def test_encoder_still_rejects_genuinely_unserialisable_types() -> None:
    with pytest.raises(TypeError):
        json.dumps({"bad": object()}, cls=_DynamoEncoder)
