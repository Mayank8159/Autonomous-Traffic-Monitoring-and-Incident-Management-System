"""
Analytics engine — pure math module for:
  • Perspective-transform-based speed estimation
  • Grid-based traffic density / congestion mapping
  • Directional vector analysis for wrong-way detection
  • IoU / deceleration / orientation-shift collision detection
  • Stationary-vehicle hazard tracking
  • Vehicle counting & flow rate
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    CFG,
    CollisionConfig,
    CongestionLevel,
    IncidentType,
    ROIConfig,
    SpeedConfig,
    StationaryConfig,
    WrongWayConfig,
)


# ---------------------------------------------------------------------------
# Lightweight data carriers
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    timestamp: float
    trajectory: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=60))
    speeds_kmh: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    first_seen: float = 0.0
    last_moving_time: float = 0.0
    is_stationary: bool = False
    bbox_angle: float = 0.0
    prev_bbox: Optional[Tuple[int, int, int, int]] = None
    prev_speed_kmh: float = 0.0
    crossed_counting_line: bool = False
    count_direction: Optional[str] = None
    lane_vector_index: int = 0


@dataclass
class Incident:
    incident_type: IncidentType
    track_id: int
    timestamp: float
    location_id: str = "intersection_01"
    frame_snapshot: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class FlowStats:
    entry_count: int = 0
    exit_count: int = 0
    flow_rate_per_min: float = 0.0
    total_vehicles: int = 0


# ---------------------------------------------------------------------------
# Perspective / homography helpers
# ---------------------------------------------------------------------------

class PerspectiveMapper:
    """Maps pixel space to real-world metric space via a homography matrix."""

    def __init__(self, src_pts: Tuple[Tuple[float, float], ...],
                 dst_pts: Tuple[Tuple[float, float], ...]) -> None:
        src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
        self._H: np.ndarray = cv2.getPerspectiveTransform(src, dst)
        self._H_inv: np.ndarray = cv2.getPerspectiveTransform(dst, src)

    def pixel_to_real(self, px: float, py: float) -> Tuple[float, float]:
        vec = np.array([[[px, py]]], dtype=np.float32)
        real = cv2.perspectiveTransform(vec, self._H)
        return float(real[0, 0, 0]), float(real[0, 0, 1])

    def pixel_distance_to_real_meters(self, p1: Tuple[float, float],
                                       p2: Tuple[float, float]) -> float:
        r1 = self.pixel_to_real(*p1)
        r2 = self.pixel_to_real(*p2)
        return math.hypot(r2[0] - r1[0], r2[1] - r1[1])


# ---------------------------------------------------------------------------
# Speed estimator  (Feature 5)
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """Estimates per-track speed via homography + temporal smoothing."""

    def __init__(self, cfg: SpeedConfig, fps: float) -> None:
        self._cfg = cfg
        self._fps = fps
        self._mapper = PerspectiveMapper(
            cfg.calibration_points_src, cfg.calibration_points_dst,
        )

    def update(self, track: TrackState, current_time: float) -> float:
        """Compute and return instantaneous speed in km/h, appended to track."""
        if len(track.trajectory) < 2:
            return 0.0
        prev = track.trajectory[-2]
        curr = track.trajectory[-1]
        dt = current_time - track.timestamp
        if dt <= 0:
            dt = 1.0 / self._fps
        real_dist_m = self._mapper.pixel_distance_to_real_meters(prev, curr)
        speed_ms = real_dist_m / dt
        speed_kmh = speed_ms * 3.6
        track.speeds_kmh.append(speed_kmh)
        if len(track.speeds_kmh) > 1:
            avg = float(np.mean(list(track.speeds_kmh)[-self._cfg.smoothing_window:]))
            return avg
        return speed_kmh

    def is_overspeed(self, speed_kmh: float) -> bool:
        return speed_kmh > self._cfg.speed_limit_kmh


# ---------------------------------------------------------------------------
# Traffic density / congestion map  (Feature 4)
# ---------------------------------------------------------------------------

class DensityAnalyzer:
    """Divides the frame into a grid and computes congestion level per cell."""

    def __init__(self, cfg: ROIConfig, frame_shape: Tuple[int, int]) -> None:
        self._rows = cfg.density_grid_rows
        self._cols = cfg.density_grid_cols
        self._h, self._w = frame_shape[:2]
        self._cell_h = self._h // self._rows
        self._cell_w = self._w // self._cols

    def compute_grid(self, bboxes: List[Tuple[int, int, int, int]]) -> List[List[float]]:
        """Return occupancy ratio [0-1] for each grid cell."""
        grid = [[0.0] * self._cols for _ in range(self._rows)]
        cell_area = self._cell_h * self._cell_w
        if cell_area == 0:
            return grid
        for x1, y1, x2, y2 in bboxes:
            for r in range(self._rows):
                for c in range(self._cols):
                    cx1 = c * self._cell_w
                    cy1 = r * self._cell_h
                    cx2 = cx1 + self._cell_w
                    cy2 = cy1 + self._cell_h
                    ix1 = max(x1, cx1)
                    iy1 = max(y1, cy1)
                    ix2 = min(x2, cx2)
                    iy2 = min(y2, cy2)
                    if ix1 < ix2 and iy1 < iy2:
                        overlap = (ix2 - ix1) * (iy2 - iy1)
                        grid[r][c] += overlap / cell_area
        return grid

    @staticmethod
    def classify_cell(occupancy: float) -> CongestionLevel:
        if occupancy < 0.15:
            return CongestionLevel.CLEAR
        if occupancy < 0.40:
            return CongestionLevel.MODERATE
        return CongestionLevel.JAMMED

    def full_density_map(self, bboxes: List[Tuple[int, int, int, int]]) -> List[List[CongestionLevel]]:
        raw = self.compute_grid(bboxes)
        return [[self.classify_cell(v) for v in row] for row in raw]


# ---------------------------------------------------------------------------
# Vehicle counter / flow measurement  (Feature 3)
# ---------------------------------------------------------------------------

class VehicleCounter:
    """Counts vehicles crossing a directional counting line (ROI)."""

    def __init__(self, cfg: ROIConfig, frame_height: int) -> None:
        self._line_start, self._line_end = cfg.counting_line
        self._direction = cfg.entry_direction
        self._frame_h = frame_height
        self._crossed: Dict[int, str] = {}
        self._entry_times: Deque[float] = deque(maxlen=300)
        self._exit_times: Deque[float] = deque(maxlen=300)
        self.stats = FlowStats()

    @staticmethod
    def _segments_intersect(p1: Tuple[int, int], p2: Tuple[int, int],
                            p3: Tuple[int, int], p4: Tuple[int, int]) -> bool:
        def _cross(o: Tuple[float, float], a: Tuple[float, float],
                   b: Tuple[float, float]) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = _cross(p3, p4, p1)
        d2 = _cross(p3, p4, p2)
        d3 = _cross(p1, p2, p3)
        d4 = _cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def update(self, track: TrackState, current_time: float) -> Optional[str]:
        """Return 'entry' / 'exit' if the track just crossed the line, else None."""
        if len(track.trajectory) < 2:
            return None
        prev = track.trajectory[-2]
        curr = track.trajectory[-1]
        if not self._segments_intersect(prev, curr, self._line_start, self._line_end):
            return None
        if track.track_id in self._crossed:
            return None

        direction: Optional[str] = None
        if self._direction == "top_to_bottom":
            direction = "entry" if curr[1] > prev[1] else "exit"
        elif self._direction == "bottom_to_top":
            direction = "entry" if curr[1] < prev[1] else "exit"
        else:
            direction = "entry"

        self._crossed[track.track_id] = direction
        if direction == "entry":
            self._entry_times.append(current_time)
            self.stats.entry_count += 1
        else:
            self._exit_times.append(current_time)
            self.stats.exit_count += 1
        self.stats.total_vehicles = self.stats.entry_count + self.stats.exit_count
        self._compute_flow_rate(current_time)
        track.crossed_counting_line = True
        track.count_direction = direction
        return direction

    def _compute_flow_rate(self, now: float) -> None:
        cutoff = now - 60.0
        recent = sum(1 for t in self._entry_times if t >= cutoff)
        self.stats.flow_rate_per_min = float(recent)


# ---------------------------------------------------------------------------
# Stationary vehicle / breakdown detector  (Feature 6)
# ---------------------------------------------------------------------------

class StationaryDetector:
    """Detects vehicles that remain nearly stationary for longer than a threshold."""

    def __init__(self, cfg: StationaryConfig) -> None:
        self._cfg = cfg
        self._last_check: Dict[int, float] = {}

    def update(self, track: TrackState, current_time: float) -> Optional[Incident]:
        speed = float(np.mean(list(track.speeds_kmh)[-5:])) if track.speeds_kmh else 0.0
        if speed < self._cfg.stationary_speed_epsilon:
            if track.first_seen == 0:
                track.first_seen = current_time
            if track.last_moving_time == 0:
                track.last_moving_time = current_time
            elapsed = current_time - track.last_moving_time
            if elapsed >= self._cfg.stationary_threshold_sec:
                track.is_stationary = True
                return Incident(
                    incident_type=IncidentType.STATIONARY_VEHICLE,
                    track_id=track.track_id,
                    timestamp=current_time,
                    metadata={
                        "stationary_duration_sec": round(elapsed, 2),
                        "class": track.class_name,
                        "bbox": track.bbox,
                        "position": track.center,
                    },
                )
        else:
            track.last_moving_time = current_time
            track.is_stationary = False
        return None


# ---------------------------------------------------------------------------
# Wrong-way driving detector  (Feature 7)
# ---------------------------------------------------------------------------

class WrongWayDetector:
    """Compares track movement vector against designated lane vectors."""

    def __init__(self, cfg: WrongWayConfig) -> None:
        self._cfg = cfg
        self._lane_vecs = [self._normalize(v) for v in cfg.lane_vectors]

    @staticmethod
    def _normalize(v: Tuple[float, float]) -> Tuple[float, float]:
        mag = math.hypot(v[0], v[1])
        return (v[0] / mag, v[1] / mag) if mag else (0.0, 0.0)

    @staticmethod
    def _angle_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dot = a[0] * b[0] + a[1] * b[1]
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def update(self, track: TrackState) -> Optional[Incident]:
        if len(track.trajectory) < self._cfg.min_track_length:
            return None
        pts = list(track.trajectory)[-self._cfg.min_track_length:]
        movement = (pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        mag = math.hypot(movement[0], movement[1])
        if mag < 5:
            return None
        mv = self._normalize(movement)
        idx = min(track.lane_vector_index, len(self._lane_vecs) - 1)
        lane = self._lane_vecs[idx]
        angle = self._angle_between(mv, lane)
        if angle > self._cfg.angle_tolerance_deg:
            return Incident(
                incident_type=IncidentType.WRONG_WAY,
                track_id=track.track_id,
                timestamp=time.time(),
                metadata={
                    "movement_vector": movement,
                    "lane_vector": lane,
                    "angle_deg": round(angle, 2),
                    "class": track.class_name,
                    "bbox": track.bbox,
                },
            )
        return None


# ---------------------------------------------------------------------------
# Collision / incident detector  (Feature 8)
# ---------------------------------------------------------------------------

class CollisionDetector:
    """Flags potential collisions via IoU spikes, deceleration, and bbox rotation."""

    def __init__(self, cfg: CollisionConfig) -> None:
        self._cfg = cfg
        self._prev_iou: Dict[Tuple[int, int], float] = {}

    @staticmethod
    def _compute_iou(a: Tuple[int, int, int, int],
                     b: Tuple[int, int, int, int]) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _bbox_angle(bbox: Tuple[int, int, int, int]) -> float:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return math.degrees(math.atan2(h, w))

    def _detect_deceleration(self, track: TrackState) -> bool:
        if len(track.speeds_kmh) < 3:
            return False
        speeds = list(track.speeds_kmh)
        recent = np.mean(speeds[-3:])
        older = np.mean(speeds[-6:-3]) if len(speeds) >= 6 else speeds[0]
        return (older - recent) > self._cfg.deceleration_threshold_kmh

    def _detect_orientation_shift(self, track: TrackState) -> bool:
        if track.prev_bbox is None:
            return False
        prev_angle = self._bbox_angle(track.prev_bbox)
        curr_angle = self._bbox_angle(track.bbox)
        return abs(curr_angle - prev_angle) > self._cfg.bbox_orientation_shift_deg

    def update(self, tracks: Dict[int, TrackState],
               current_time: float) -> List[Incident]:
        incidents: List[Incident] = []
        track_list = list(tracks.values())

        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                t_a, t_b = track_list[i], track_list[j]
                iou = self._compute_iou(t_a.bbox, t_b.bbox)
                key = (min(t_a.track_id, t_b.track_id),
                       max(t_a.track_id, t_b.track_id))
                prev = self._prev_iou.get(key, 0.0)
                self._prev_iou[key] = iou

                if iou > self._cfg.iou_spike_threshold and \
                   (iou - prev) > self._cfg.iou_spike_delta:
                    incidents.append(Incident(
                        incident_type=IncidentType.COLLISION,
                        track_id=t_a.track_id,
                        timestamp=current_time,
                        metadata={
                            "iou": round(iou, 4),
                            "iou_delta": round(iou - prev, 4),
                            "track_b": t_b.track_id,
                            "position_a": t_a.center,
                            "position_b": t_b.center,
                        },
                    ))

        for t in track_list:
            if self._detect_deceleration(t):
                incidents.append(Incident(
                    incident_type=IncidentType.COLLISION,
                    track_id=t.track_id,
                    timestamp=current_time,
                    metadata={
                        "cause": "sudden_deceleration",
                        "speeds": [round(s, 1) for s in t.speeds_kmh],
                    },
                ))
            if self._detect_orientation_shift(t):
                incidents.append(Incident(
                    incident_type=IncidentType.COLLISION,
                    track_id=t.track_id,
                    timestamp=current_time,
                    metadata={
                        "cause": "bbox_orientation_shift",
                        "prev_bbox": t.prev_bbox,
                        "curr_bbox": t.bbox,
                    },
                ))
        return incidents


# ---------------------------------------------------------------------------
# Composite analytics coordinator
# ---------------------------------------------------------------------------

class AnalyticsEngine:
    """Single façade that owns every sub-analyser and exposes a unified update()."""

    def __init__(self, frame_w: int, frame_h: int, fps: float) -> None:
        self.speed = SpeedEstimator(CFG.speed, fps)
        self.density = DensityAnalyzer(CFG.roi, (frame_h, frame_w))
        self.counter = VehicleCounter(CFG.roi, frame_h)
        self.stationary = StationaryDetector(CFG.stationary)
        self.wrong_way = WrongWayDetector(CFG.wrong_way)
        self.collision = CollisionDetector(CFG.collision)
        self.tracks: Dict[int, TrackState] = {}
        self.incidents: List[Incident] = []
        self.frame_count: int = 0

    def update_tracks(self, detections: List[Tuple[int, int, float, float, float, int]],
                      current_time: float) -> None:
        """
        Feed detections: [(track_id, class_id, x1, y1, x2, y2, conf), ...]
        """
        self.frame_count += 1
        active_ids = set()
        for tid, cls_id, x1, y1, x2, y2, conf in detections:
            active_ids.add(tid)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if tid not in self.tracks:
                self.tracks[tid] = TrackState(
                    track_id=tid,
                    class_name=CFGLookup.class_name(cls_id),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(cx, cy),
                    confidence=conf,
                    timestamp=current_time,
                    first_seen=current_time,
                    last_moving_time=current_time,
                )
            t = self.tracks[tid]
            t.prev_bbox = t.bbox
            t.bbox = (int(x1), int(y1), int(x2), int(y2))
            t.center = (cx, cy)
            t.confidence = conf
            t.timestamp = current_time
            t.trajectory.append((cx, cy))

        # Purge stale tracks
        stale = [tid for tid in self.tracks if tid not in active_ids]
        for tid in stale:
            del self.tracks[tid]

    def run_analytics(self, current_time: float) -> List[Incident]:
        """Execute all analytics checks; return new incidents detected this frame."""
        new_incidents: List[Incident] = []
        for t in self.tracks.values():
            spd = self.speed.update(t, current_time)
            if self.speed.is_overspeed(spd):
                new_incidents.append(Incident(
                    incident_type=IncidentType.OVERSPEED,
                    track_id=t.track_id,
                    timestamp=current_time,
                    metadata={
                        "speed_kmh": round(spd, 1),
                        "limit_kmh": CFG.speed.speed_limit_kmh,
                        "class": t.class_name,
                    },
                ))
            self.counter.update(t, current_time)
            s_inc = self.stationary.update(t, current_time)
            if s_inc:
                new_incidents.append(s_inc)
            w_inc = self.wrong_way.update(t)
            if w_inc:
                new_incidents.append(w_inc)
        col_inc = self.collision.update(self.tracks, current_time)
        new_incidents.extend(col_inc)
        self.incidents.extend(new_incidents)
        return new_incidents

    def get_density_map(self) -> List[List[CongestionLevel]]:
        bboxes = [t.bbox for t in self.tracks.values()]
        return self.density.full_density_map(bboxes)

    def get_flow_stats(self) -> FlowStats:
        return self.counter.stats


class CFGLookup:
    @staticmethod
    def class_name(cls_id: int) -> str:
        from config import VEHICLE_CLASSES
        return VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
