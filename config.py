"""
Central configuration for the Autonomous Traffic Monitoring & Incident Management System.
All tunables, thresholds, ROI definitions, and stream parameters live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CongestionLevel(Enum):
    CLEAR = "clear"
    MODERATE = "moderate"
    JAMMED = "jammed"


class IncidentType(Enum):
    STATIONARY_VEHICLE = "stationary_vehicle"
    WRONG_WAY = "wrong_way"
    COLLISION = "collision"
    OVERSPEED = "overspeed"


# ---------------------------------------------------------------------------
# Vehicle class labels shipped with COCO-trained YOLOv8
# ---------------------------------------------------------------------------

VEHICLE_CLASSES: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ---------------------------------------------------------------------------
# Data classes for clean config grouping
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    source: str = "0"
    rtsp_transport: str = "tcp"
    frame_width: int = 1280
    frame_height: int = 720
    buffer_size: int = 64
    reconnect_delay: float = 3.0


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.5
    device: str = "0"
    half_precision: bool = True
    max_detections: int = 300
    classes: Tuple[int, ...] = (1, 2, 3, 5, 7)


@dataclass
class TrackerConfig:
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 60
    match_thresh: float = 0.8
    min_hits: int = 3
    frame_rate: int = 30


@dataclass
class ROIConfig:
    counting_line: Tuple[Tuple[int, int], Tuple[int, int]] = ((100, 500), (1100, 500))
    entry_direction: str = "top_to_bottom"
    density_grid_rows: int = 3
    density_grid_cols: int = 3


@dataclass
class SpeedConfig:
    calibration_points_src: Tuple[Tuple[float, float], ...] = (
        (200, 500), (600, 500), (200, 700), (600, 700),
    )
    calibration_points_dst: Tuple[Tuple[float, float], ...] = (
        (0, 0), (20, 0), (0, 20), (20, 20),
    )
    pixels_to_meters_ratio: float = 0.05
    speed_limit_kmh: float = 60.0
    smoothing_window: int = 5


@dataclass
class StationaryConfig:
    stationary_threshold_sec: float = 5.0
    stationary_speed_epsilon: float = 2.0
    check_interval_frames: int = 10


@dataclass
class WrongWayConfig:
    lane_vectors: Tuple[Tuple[float, float], ...] = (
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    )
    angle_tolerance_deg: float = 90.0
    min_track_length: int = 10


@dataclass
class CollisionConfig:
    iou_spike_threshold: float = 0.3
    iou_spike_delta: float = 0.15
    deceleration_threshold_kmh: float = 30.0
    bbox_orientation_shift_deg: float = 30.0
    pre_collision_frames: int = 15


@dataclass
class AlertConfig:
    webhook_urls: Tuple[str, ...] = ()
    snapshot_dir: str = "snapshots"
    log_dir: str = "logs"
    max_concurrent_dispatches: int = 10
    snapshot_quality: int = 90
    enable_sms_mock: bool = True


@dataclass
class VisualizationConfig:
    show_bounding_boxes: bool = True
    show_trajectories: bool = True
    show_speed_vectors: bool = True
    show_density_heatmap: bool = True
    show_counting_line: bool = True
    show_alerts: bool = True
    trajectory_length: int = 30
    font_scale: float = 0.6
    font_thickness: int = 2
    bbox_thickness: int = 2
    clear_color: Tuple[int, int, int] = (0, 200, 0)
    moderate_color: Tuple[int, int, int] = (0, 200, 255)
    jammed_color: Tuple[int, int, int] = (0, 0, 255)
    alert_color: Tuple[int, int, int] = (0, 0, 255)


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    enable_docs: bool = True


# ---------------------------------------------------------------------------
# Master config combining every sub-config
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    stationary: StationaryConfig = field(default_factory=StationaryConfig)
    wrong_way: WrongWayConfig = field(default_factory=WrongWayConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global singleton
CFG = SystemConfig()
