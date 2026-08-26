"""Lambda handler: Analytics processing -- compute stats from stored detections."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import CFG, CongestionLevel
from app.utils import get_dynamodb_resource, ok, error

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: dict[str, Any], context: Any) -> dict:
    """Compute analytics from recent detections in DynamoDB."""
    try:
        dynamodb = get_dynamodb_resource()
        table = dynamodb.Table(CFG.aws.dynamodb_table_name)

        resp = table.scan(Limit=500)
        items = resp.get("Items", [])

        tracks = {}
        for item in items:
            tid = item.get("track_id", 0)
            bbox = item.get("bbox", [0, 0, 0, 0])
            center = item.get("center", [0, 0])
            tracks[tid] = {
                "track_id": tid,
                "class_name": item.get("class_name", "unknown"),
                "bbox": bbox,
                "center": center,
                "confidence": item.get("confidence", 0.0),
                "timestamp": item.get("timestamp", 0),
            }

        density_grid = _compute_density(tracks, rows=3, cols=3)

        return ok({
            "active_tracks": len(tracks),
            "density_map": density_grid,
        })

    except Exception as exc:
        logger.error("Analytics failed: %s", exc)
        return error(500, str(exc))


def _compute_density(tracks: dict, rows: int = 3, cols: int = 3) -> list:
    grid = [[0.0] * cols for _ in range(rows)]
    for t in tracks.values():
        bbox = t.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            r = min(int(cy / 240), rows - 1)
            c = min(int(cx / 426), cols - 1)
            grid[max(0, r)][max(0, c)] += 1

    result = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            level = "clear" if v < 2 else "moderate" if v < 5 else "jammed"
            result.append({"row": r, "col": c, "level": level})
    return result
