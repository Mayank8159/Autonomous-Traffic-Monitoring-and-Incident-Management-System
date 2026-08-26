"""Lambda handler: API endpoints for live telemetry data."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import CFG
from app.utils import get_dynamodb_resource, ok, error, query_tracks, query_incidents

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: dict[str, Any], context: Any) -> dict:
    """Route API requests based on path."""
    path = event.get("path", "/")
    method = event.get("httpMethod", "GET")

    if method == "OPTIONS":
        return ok({"message": "OK"})

    routes = {
        "/api/tracks": _get_tracks,
        "/api/flow": _get_flow,
        "/api/density": _get_density,
        "/api/incidents": _get_incidents,
        "/api/status": _get_status,
        "/api/config": _get_config,
    }

    route_fn = routes.get(path)
    if route_fn:
        try:
            return route_fn(event)
        except Exception as exc:
            logger.error("Route %s failed: %s", path, exc)
            return error(500, str(exc))

    return error(404, f"Not found: {path}")


def _get_tracks(event: dict) -> dict:
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(CFG.aws.dynamodb_table_name)

    resp = table.scan(Limit=200)
    items = resp.get("Items", [])

    tracks = []
    for item in items:
        bbox = item.get("bbox", [0, 0, 0, 0])
        center = item.get("center", [0, 0])
        tracks.append({
            "track_id": item.get("track_id", 0),
            "class_name": item.get("class_name", "unknown"),
            "bbox": bbox,
            "center": center,
            "speed_kmh": 0.0,
            "is_stationary": False,
            "confidence": float(item.get("confidence", 0.0)),
            "timestamp": float(item.get("timestamp", 0)),
        })

    return ok(tracks)


def _get_flow(event: dict) -> dict:
    return ok({
        "entry_count": 0,
        "exit_count": 0,
        "total_vehicles": 0,
        "flow_rate_per_min": 0.0,
    })


def _get_density(event: dict) -> dict:
    return ok([
        {"row": 0, "col": 0, "level": "clear"},
        {"row": 0, "col": 1, "level": "clear"},
        {"row": 0, "col": 2, "level": "clear"},
        {"row": 1, "col": 0, "level": "clear"},
        {"row": 1, "col": 1, "level": "clear"},
        {"row": 1, "col": 2, "level": "clear"},
        {"row": 2, "col": 0, "level": "clear"},
        {"row": 2, "col": 1, "level": "clear"},
        {"row": 2, "col": 2, "level": "clear"},
    ])


def _get_incidents(event: dict) -> dict:
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(CFG.aws.incident_table_name)

    params = event.get("queryStringParameters") or {}
    incident_type = params.get("type")
    limit = int(params.get("limit", 20))

    items = query_incidents(table, incident_type, limit)

    incidents = []
    for item in items:
        incidents.append({
            "incident_id": item.get("incident_id", ""),
            "timestamp": float(item.get("timestamp", 0)),
            "type": item.get("incident_type", "unknown"),
            "track_id": item.get("track_id", -1),
            "location_id": item.get("location_id", "unknown"),
            "metadata": item.get("metadata", {}),
            "snapshot_path": item.get("snapshot_path"),
        })

    return ok(incidents)


def _get_status(event: dict) -> dict:
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(CFG.aws.incident_table_name)
    resp = table.scan(Limit=1000)
    total_incidents = resp.get("Count", 0)

    return ok({
        "active_tracks": 0,
        "total_incidents": total_incidents,
        "last_update": time.time() * 1000,
    })


def _get_config(event: dict) -> dict:
    return ok({
        "model": CFG.detection.model_name,
        "confidence_threshold": CFG.detection.confidence_threshold,
        "speed_limit_kmh": CFG.speed.speed_limit_kmh,
        "stationary_threshold_sec": CFG.stationary.stationary_threshold_sec,
        "device": CFG.detection.device,
    })
