"""AWS Lambda entry point serving precomputed clothing-fit predictions.

Deliberately dependency-free (standard library + boto3, which the Lambda
runtime already provides) so the deployment package stays a single file and
needs no build step or layer.

The ordinal model and conformal calibration are not run here.
``scripts/publish_predictions.py`` (run after ``clothing-fit train`` and
``clothing-fit evaluate``) writes the precomputed payloads to DynamoDB, so
this handler is a key lookup plus a response envelope -- the same
"precompute, don't infer on the request path" design as a top-N
recommendation service.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any

try:  # pragma: no cover - boto3 is provided by the Lambda runtime.
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None

PREDICTIONS_TABLE = os.environ.get("PREDICTIONS_TABLE", "clothing-fit-dev-predictions")

_table = None


class _DynamoEncoder(json.JSONEncoder):
    """Serialize the ``Decimal`` values DynamoDB returns for every number.

    boto3's resource API deserializes DynamoDB's ``N`` type to ``Decimal``,
    which ``json.dumps`` rejects outright. Writing the payload works fine,
    so this only ever fails on the read path -- the first real invocation,
    not in any test that checks the pre-write payload.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return int(o) if o == o.to_integral_value() else float(o)
        return super().default(o)


def _get_table():
    """Create the DynamoDB resource once per container, not per request."""
    global _table
    if _table is None:
        if boto3 is None:
            raise RuntimeError("boto3 is unavailable; cannot reach DynamoDB.")
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        resource = (
            boto3.resource("dynamodb", endpoint_url=endpoint)
            if endpoint
            else boto3.resource("dynamodb")
        )
        _table = resource.Table(PREDICTIONS_TABLE)
    return _table


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, cls=_DynamoEncoder),
    }


def parse_request(event: dict[str, Any]) -> str:
    """Extract and validate ``request_id`` from an API Gateway event.

    Split out from the handler so the validation rule is unit-testable
    without any AWS dependency.

    Args:
        event: API Gateway proxy event, or a bare dict for direct invocation.

    Returns:
        The composite ``request_id`` (``"{user_id}#{item_id}#{size}"``).

    Raises:
        ValueError: If ``request_id`` is missing or empty.
    """
    params = event.get("queryStringParameters") or {}
    if not params and "request_id" in event:
        params = event

    req_id = params.get("request_id")
    if req_id in (None, ""):
        raise ValueError("Query parameter 'request_id' is required.")
    return str(req_id)


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:  # noqa: ARG001
    """Return the stored prediction for one ``(user, item, size)`` request."""
    try:
        req_id = parse_request(event)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    try:
        item = _get_table().get_item(Key={"request_id": req_id}).get("Item")
    except Exception as exc:  # noqa: BLE001 - surface as 503, never leak a stack trace.
        return _response(503, {"error": "prediction store unavailable", "detail": str(exc)})

    if not item:
        return _response(
            404,
            {"error": f"no prediction stored for request {req_id}", "request_id": req_id},
        )

    return _response(200, item)
