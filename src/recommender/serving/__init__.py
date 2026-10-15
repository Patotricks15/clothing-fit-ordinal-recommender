"""Serving infrastructure: precomputed DynamoDB payloads + Lambda handler."""

from __future__ import annotations

from recommender.serving.handler import lambda_handler, parse_request
from recommender.serving.predictions import build_prediction_payloads, request_id

__all__ = [
    "build_prediction_payloads",
    "lambda_handler",
    "parse_request",
    "request_id",
]
