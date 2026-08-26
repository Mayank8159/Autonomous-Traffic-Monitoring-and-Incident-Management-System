"""AWS Lambda utility functions for responses, S3, and DynamoDB."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

import boto3
from botocore.config import Config

from app.config import CFG

logger = logging.getLogger(__name__)

_boto_config = Config(region_name=CFG.aws.region, retries={"max_attempts": 3, "mode": "adaptive"})


def get_dynamodb_resource():
    return boto3.resource("dynamodb", config=_boto_config)


def get_s3_client():
    return boto3.client("s3", config=_boto_config)


def lambda_response(status_code: int, body: Any) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }


def ok(body: Any) -> dict:
    return lambda_response(200, body)


def error(status_code: int, message: str) -> dict:
    return lambda_response(status_code, {"error": message})


def generate_id() -> str:
    return str(uuid.uuid4())[:8]


def now_ms() -> float:
    return time.time() * 1000


def store_track(table, track_data: dict) -> None:
    table.put_item(Item=track_data)


def store_incident(table, incident_data: dict) -> None:
    table.put_item(Item=incident_data)


def query_tracks(table, pk: str, limit: int = 100) -> list[dict]:
    resp = table.query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": pk},
        Limit=limit,
        ScanIndexForward=False,
    )
    return resp.get("Items", [])


def query_incidents(table, incident_type: Optional[str] = None, limit: int = 20) -> list[dict]:
    if incident_type:
        resp = table.query(
            IndexName="type-timestamp-index",
            KeyConditionExpression="incident_type = :t",
            ExpressionAttributeValues={":t": incident_type},
            Limit=limit,
            ScanIndexForward=False,
        )
    else:
        resp = table.scan(Limit=limit)
    return resp.get("Items", [])
