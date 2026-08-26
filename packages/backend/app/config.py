"""Central configuration for the serverless Traffic Monitoring System."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class CongestionLevel(Enum):
    CLEAR = "clear"
    MODERATE = "moderate"
    JAMMED = "jammed"


class IncidentType(Enum):
    STATIONARY_VEHICLE = "stationary_vehicle"
    WRONG_WAY = "wrong_way"
    COLLISION = "collision"
    OVERSPEED = "overspeed"


VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.5
    device: str = "cpu"
    half_precision: bool = False
    max_detections: int = 300
    classes: tuple[int, ...] = (1, 2, 3, 5, 7)


@dataclass
class TrackerConfig:
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 60
    match_thresh: float = 0.8
    min_hits: int = 3
    frame_rate: int = 30


@dataclass
class SpeedConfig:
    pixels_to_meters_ratio: float = 0.05
    speed_limit_kmh: float = 60.0
    smoothing_window: int = 5


@dataclass
class StationaryConfig:
    stationary_threshold_sec: float = 5.0
    stationary_speed_epsilon: float = 2.0


@dataclass
class WrongWayConfig:
    angle_tolerance_deg: float = 90.0
    min_track_length: int = 10


@dataclass
class CollisionConfig:
    iou_spike_threshold: float = 0.3
    iou_spike_delta: float = 0.15
    deceleration_threshold_kmh: float = 30.0


@dataclass
class AWSConfig:
    dynamodb_table_name: str = field(
        default_factory=lambda: os.environ.get("DYNAMODB_TABLE_NAME", "traffic-monitor-dev")
    )
    incident_table_name: str = field(
        default_factory=lambda: os.environ.get("INCIDENT_TABLE_NAME", "traffic-incidents-dev")
    )
    snapshot_bucket: str = field(
        default_factory=lambda: os.environ.get("SNAPSHOT_BUCKET", "traffic-snapshots-dev")
    )
    region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1")
    )


@dataclass
class SystemConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    stationary: StationaryConfig = field(default_factory=StationaryConfig)
    wrong_way: WrongWayConfig = field(default_factory=WrongWayConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)


CFG = SystemConfig()
