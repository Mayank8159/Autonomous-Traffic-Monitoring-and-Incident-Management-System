"""Lambda handler: Ingest detection results from edge YOLOv8 processor."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import CFG
from app.utils import get_dynamodb_resource, generate_id, store_track, ok, error

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: dict[str, Any], context: Any) -> dict:
    """Accept POST with detection results and store in DynamoDB."""
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return error(400, "Invalid JSON")

    detections = body.get("detections", [])
    frame_id = body.get("frame_id", generate_id())
    timestamp_ms = body.get("timestamp", int(time.time() * 1000))

    if not detections:
        return ok({"stored": 0, "frame_id": frame_id})

    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(CFG.aws.dynamodb_table_name)

    count = 0
    for det in detections:
        track_data = {
            "pk": f"FRAME#{frame_id}",
            "sk": f"TRACK#{det.get('track_id', 0):06d}",
            "track_id": det.get("track_id", 0),
            "class_id": det.get("class_id", 0),
            "class_name": det.get("class_name", "unknown"),
            "bbox": det.get("bbox", [0, 0, 0, 0]),
            "center": det.get("center", [0, 0]),
            "confidence": det.get("confidence", 0.0),
            "timestamp": timestamp_ms,
        }
        store_track(table, track_data)
        count += 1

    logger.info("Stored %d detections for frame %s", count, frame_id)
    return ok({"stored": count, "frame_id": frame_id})
