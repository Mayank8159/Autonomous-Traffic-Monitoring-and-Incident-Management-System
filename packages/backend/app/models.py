"""Shared Pydantic models for request/response serialization."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class TrackInfo(BaseModel):
    track_id: int
    class_name: str
    bbox: list[int]
    center: list[int]
    speed_kmh: float
    is_stationary: bool
    confidence: float
    timestamp: float


class FlowStats(BaseModel):
    entry_count: int
    exit_count: int
    total_vehicles: int
    flow_rate_per_min: float


class DensityCell(BaseModel):
    row: int
    col: int
    level: str


class IncidentInfo(BaseModel):
    incident_id: str
    timestamp: float
    type: str
    track_id: int
    location_id: str
    metadata: dict[str, Any]
    snapshot_path: Optional[str] = None


class SystemStatus(BaseModel):
    active_tracks: int
    total_incidents: int
    last_update: float


class DetectionResult(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    bbox: list[float]
    confidence: float


class FrameUploadEvent(BaseModel):
    s3_key: str
    bucket: str
    timestamp: float
    frame_width: int = 1280
    frame_height: int = 720
