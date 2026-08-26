"""Lambda handler: Alert dispatch -- webhooks, SMS, structured logging."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from app.config import CFG
from app.utils import get_dynamodb_resource, get_s3_client, generate_id

logger = logging.getLogger()
logger.setLevel(logging.INFO)

WEBHOOK_URLS = [u.strip() for u in os.environ.get("WEBHOOK_URLS", "").split(",") if u.strip()]


def handler(event: dict[str, Any], context: Any) -> dict:
    """Handle alert dispatch from DynamoDB Stream or direct invocation."""
    dynamodb = get_dynamodb_resource()
    incident_table = dynamodb.Table(CFG.aws.incident_table_name)

    records = event.get("Records", [])

    if records:
        for record in records:
            if record.get("eventName") in ("INSERT", "MODIFY"):
                _handle_dynamodb_record(record)
    else:
        _handle_direct_invocation(event, incident_table)

    return {"statusCode": 200, "body": json.dumps({"status": "alerts_processed"})}


def _handle_dynamodb_record(record: dict) -> None:
    new_image = record.get("dynamodb", {}).get("NewImage", {})
    if not new_image:
        return

    incident = _deserialize_dynamo(new_image)
    incident_type = incident.get("incident_type", "unknown")
    track_id = incident.get("track_id", -1)

    if incident_type in ("collision", "wrong_way"):
        _send_mock_sms(incident_type, track_id)

    if WEBHOOK_URLS:
        _dispatch_webhooks(incident)

    logger.info("Alert dispatched: %s (track %s)", incident_type, track_id)


def _handle_direct_invocation(event: dict, table: Any) -> None:
    incident_type = event.get("incident_type", "collision")
    track_id = event.get("track_id", 0)

    incident_id = f"INC-{generate_id()}"
    table.put_item(Item={
        "incident_id": incident_id,
        "timestamp": int(time.time() * 1000),
        "incident_type": incident_type,
        "track_id": track_id,
        "location_id": event.get("location_id", "intersection_01"),
        "metadata": event.get("metadata", {}),
    })


def _send_mock_sms(incident_type: str, track_id: int) -> None:
    body = f"ALERT: {incident_type} detected (track {track_id}) at intersection_01"
    logger.info("[SMS MOCK] To=+0000000000  Body=%s", body)


def _dispatch_webhooks(payload: dict) -> None:
    import urllib.request
    data = json.dumps(payload, default=str).encode()
    for url in WEBHOOK_URLS:
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            logger.warning("Webhook %s failed: %s", url, exc)


def _deserialize_dynamo(image: dict) -> dict:
    result = {}
    for key, val in image.items():
        if "S" in val:
            result[key] = val["S"]
        elif "N" in val:
            result[key] = float(val["N"])
        elif "BOOL" in val:
            result[key] = val["BOOL"]
    return result
