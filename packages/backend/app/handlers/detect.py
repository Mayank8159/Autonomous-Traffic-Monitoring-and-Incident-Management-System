"""Lambda handler: YOLOv8 vehicle detection on S3-uploaded frames."""

from __future__ import annotations

import json
import logging
import time
from io import BytesIO
from typing import Any

import cv2
import numpy as np

from app.config import CFG
from app.detector import Detector
from app.utils import get_dynamodb_resource, get_s3_client, generate_id, store_track

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_detector: Detector | None = None


def _get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def handler(event: dict[str, Any], context: Any) -> dict:
    """Handle S3 event triggers -- run detection on uploaded frames."""
    from app.config import CFG

    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(CFG.aws.dynamodb_table_name)
    s3 = get_s3_client()

    records = event.get("Records", [])
    if not records:
        return {"statusCode": 200, "body": "No records"}

    for record in records:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        try:
            _process_frame(s3, table, bucket, key)
        except Exception as exc:
            logger.error("Detection failed for %s/%s: %s", bucket, key, exc)
            return {"statusCode": 500, "body": str(exc)}

    return {"statusCode": 200, "body": json.dumps({"processed": len(records)})}


def _process_frame(s3: Any, table: Any, bucket: str, key: str) -> None:
    resp = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = resp["Body"].read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        logger.warning("Failed to decode frame: %s", key)
        return

    detector = _get_detector()
    detections = detector.detect(frame)
    now = time.time()

    timestamp_ms = int(now * 1000)
    frame_id = generate_id()

    for track_id, cls_id, x1, y1, x2, y2, conf in detections:
        track_data = {
            "pk": f"FRAME#{frame_id}",
            "sk": f"TRACK#{track_id:06d}",
            "track_id": track_id,
            "class_id": cls_id,
            "class_name": detector.class_name(cls_id),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "confidence": round(conf, 4),
            "timestamp": timestamp_ms,
            "s3_key": key,
        }
        store_track(table, track_data)

    logger.info("Frame %s: %d detections", frame_id, len(detections))
