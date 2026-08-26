"""Lambda handler: Analytics processing -- speed, density, collision, wrong-way detection."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.analytics import AnalyticsEngine
from app.config import CFG
from app.utils import get_dynamodb_resource, get_s3_client, store_incident, generate_id

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_analytics: AnalyticsEngine | None = None


def _get_analytics() -> AnalyticsEngine:
    global _analytics
    if _analytics is None:
        _analytics = AnalyticsEngine()
    return _analytics


def handler(event: dict[str, Any], context: Any) -> dict:
    """Scheduled trigger -- process analytics from recent detections."""
    dynamodb = get_dynamodb_resource()
    track_table = dynamodb.Table(CFG.aws.dynamodb_table_name)
    incident_table = dynamodb.Table(CFG.aws.incident_table_name)

    now = time.time()
    analytics = _get_analytics()

    try:
        resp = track_table.scan(Limit=500)
        items = resp.get("Items", [])

        detections = []
        for item in items:
            bbox = item.get("bbox", [0, 0, 0, 0])
            detections.append((
                item.get("track_id", 0),
                item.get("class_id", 0),
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(item.get("confidence", 0.0)),
            ))

        analytics.update_tracks(detections, now)
        new_incidents = analytics.run_analytics(now)

        for inc in new_incidents:
            incident_id = f"INC-{generate_id()}"
            incident_data = {
                "incident_id": incident_id,
                "timestamp": int(inc.timestamp * 1000),
                "incident_type": inc.incident_type.value,
                "track_id": inc.track_id,
                "location_id": inc.location_id,
                "metadata": inc.metadata,
            }
            store_incident(incident_table, incident_data)

        density_map = analytics.get_density_map()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "processed_tracks": len(detections),
                "new_incidents": len(new_incidents),
                "density_map": [[cell.value for cell in row] for row in density_map],
            }),
        }

    except Exception as exc:
        logger.error("Analytics processing failed: %s", exc)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
