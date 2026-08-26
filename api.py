"""
Lightweight FastAPI server exposing live telemetry as JSON.

Run standalone:  ``uvicorn api:app --host 0.0.0.0 --port 8000``
Run alongside main.py:  import api in a background thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analytics import AnalyticsEngine, CongestionLevel, FlowStats
from config import CFG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (populated by main loop)
# ---------------------------------------------------------------------------

_analytics: Optional[AnalyticsEngine] = None
_alert_recent: Optional[List[Dict[str, Any]]] = None


def bind_analytics(engine: AnalyticsEngine,
                   recent_alerts: List[Dict[str, Any]]) -> None:
    """Called once from the main loop to share state with the API."""
    global _analytics, _alert_recent
    _analytics = engine
    _alert_recent = recent_alerts


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class TrackInfo(BaseModel):
    track_id: int
    class_name: str
    bbox: List[int]
    center: List[int]
    speed_kmh: float
    is_stationary: bool
    confidence: float


class FlowInfo(BaseModel):
    entry_count: int
    exit_count: int
    total_vehicles: int
    flow_rate_per_min: float


class DensityCell(BaseModel):
    row: int
    col: int
    level: str


class IncidentInfo(BaseModel):
    timestamp: float
    type: str
    track_id: int
    location_id: str
    metadata: Dict[str, Any]
    snapshot_path: Optional[str] = None


class SystemStatus(BaseModel):
    active_tracks: int
    total_incidents: int
    fps: float
    frame_count: int


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Traffic Monitor — Live Telemetry API",
    version="1.0.0",
    docs_url="/docs" if CFG.api.enable_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "Autonomous Traffic Monitor"}


@app.get("/health", tags=["health"])
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/tracks", response_model=List[TrackInfo], tags=["telemetry"])
def get_tracks() -> List[TrackInfo]:
    """Return all currently tracked vehicles."""
    _ensure_ready()
    assert _analytics is not None
    result: List[TrackInfo] = []
    for t in _analytics.tracks.values():
        spd = float(t.speeds_kmh[-1]) if t.speeds_kmh else 0.0
        result.append(TrackInfo(
            track_id=t.track_id,
            class_name=t.class_name,
            bbox=list(t.bbox),
            center=list(t.center),
            speed_kmh=round(spd, 1),
            is_stationary=t.is_stationary,
            confidence=round(t.confidence, 3),
        ))
    return result


@app.get("/api/flow", response_model=FlowInfo, tags=["telemetry"])
def get_flow() -> FlowInfo:
    """Return live vehicle flow statistics."""
    _ensure_ready()
    assert _analytics is not None
    fs = _analytics.get_flow_stats()
    return FlowInfo(
        entry_count=fs.entry_count,
        exit_count=fs.exit_count,
        total_vehicles=fs.total_vehicles,
        flow_rate_per_min=round(fs.flow_rate_per_min, 2),
    )


@app.get("/api/density", response_model=List[DensityCell], tags=["telemetry"])
def get_density() -> List[DensityCell]:
    """Return grid-based congestion map."""
    _ensure_ready()
    assert _analytics is not None
    grid = _analytics.get_density_map()
    cells: List[DensityCell] = []
    for r, row in enumerate(grid):
        for c, level in enumerate(row):
            cells.append(DensityCell(row=r, col=c, level=level.value))
    return cells


@app.get("/api/incidents", response_model=List[IncidentInfo], tags=["alerts"])
def get_incidents(limit: int = 20) -> List[IncidentInfo]:
    """Return the most recent incidents/alerts."""
    _ensure_ready()
    assert _alert_recent is not None
    result: List[IncidentInfo] = []
    for entry in _alert_recent[-limit:]:
        result.append(IncidentInfo(
            timestamp=entry.get("timestamp", 0.0),
            type=entry.get("type", "unknown"),
            track_id=entry.get("track_id", -1),
            location_id=entry.get("location_id", "unknown"),
            metadata=entry.get("metadata", {}),
            snapshot_path=entry.get("snapshot_path"),
        ))
    return result


@app.get("/api/status", response_model=SystemStatus, tags=["telemetry"])
def get_status() -> SystemStatus:
    """Return overall system status."""
    _ensure_ready()
    assert _analytics is not None
    return SystemStatus(
        active_tracks=len(_analytics.tracks),
        total_incidents=len(_analytics.incidents),
        fps=0.0,
        frame_count=_analytics.frame_count,
    )


@app.get("/api/config", tags=["debug"])
def get_config() -> Dict[str, Any]:
    """Return non-sensitive configuration values."""
    return {
        "model": CFG.detection.model_name,
        "confidence_threshold": CFG.detection.confidence_threshold,
        "speed_limit_kmh": CFG.speed.speed_limit_kmh,
        "stationary_threshold_sec": CFG.stationary.stationary_threshold_sec,
        "stream_source": CFG.stream.source,
        "device": CFG.detection.device,
    }


def _ensure_ready() -> None:
    if _analytics is None:
        raise HTTPException(
            status_code=503,
            detail="Analytics engine not yet initialised. Start main.py first.",
        )


# ---------------------------------------------------------------------------
# Background runner (for embedding inside main.py's event loop)
# ---------------------------------------------------------------------------

def start_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch uvicorn in a daemon thread."""
    import threading

    def _run() -> None:
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="api-server")
    t.start()
    logger.info("API server started on http://%s:%d", host, port)
