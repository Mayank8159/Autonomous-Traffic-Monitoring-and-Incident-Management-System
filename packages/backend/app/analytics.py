"""Analytics engine -- speed estimation, density mapping, collision detection, wrong-way detection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from app.config import CFG, CongestionLevel, IncidentType


@dataclass
class TrackState:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float
    timestamp: float
    trajectory: Deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=60))
    speeds_kmh: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    first_seen: float = 0.0
    last_moving_time: float = 0.0
    is_stationary: bool = False
    prev_bbox: Optional[tuple[int, int, int, int]] = None


@dataclass
class Incident:
    incident_type: IncidentType
    track_id: int
    timestamp: float
    location_id: str = "intersection_01"
    metadata: dict = field(default_factory=dict)


@dataclass
class FlowStats:
    entry_count: int = 0
    exit_count: int = 0
    flow_rate_per_min: float = 0.0
    total_vehicles: int = 0


class SpeedEstimator:
    def __init__(self) -> None:
        self._cfg = CFG.speed

    def update(self, track: TrackState, current_time: float) -> float:
        if len(track.trajectory) < 2:
            return 0.0
        prev, curr = track.trajectory[-2], track.trajectory[-1]
        dt = current_time - track.timestamp
        if dt <= 0:
            dt = 1.0 / 30.0
        dist_px = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        dist_m = dist_px * self._cfg.pixels_to_meters_ratio
        speed_kmh = (dist_m / dt) * 3.6
        track.speeds_kmh.append(speed_kmh)
        if len(track.speeds_kmh) > 1:
            return float(np.mean(list(track.speeds_kmh)[-self._cfg.smoothing_window:]))
        return speed_kmh

    def is_overspeed(self, speed_kmh: float) -> bool:
        return speed_kmh > self._cfg.speed_limit_kmh


class DensityAnalyzer:
    def __init__(self, rows: int = 3, cols: int = 3) -> None:
        self._rows, self._cols = rows, cols

    def full_density_map(self, bboxes: list[tuple[int, int, int, int]],
                         frame_shape: tuple[int, int]) -> list[list[CongestionLevel]]:
        h, w = frame_shape
        cell_h, cell_w = h // self._rows, w // self._cols
        cell_area = cell_h * cell_w
        grid = [[0.0] * self._cols for _ in range(self._rows)]

        for x1, y1, x2, y2 in bboxes:
            for r in range(self._rows):
                for c in range(self._cols):
                    cx1, cy1 = c * cell_w, r * cell_h
                    cx2, cy2 = cx1 + cell_w, cy1 + cell_h
                    ix1, iy1 = max(x1, cx1), max(y1, cy1)
                    ix2, iy2 = min(x2, cx2), min(y2, cy2)
                    if ix1 < ix2 and iy1 < iy2:
                        grid[r][c] += (ix2 - ix1) * (iy2 - iy1) / cell_area

        return [
            [
                CongestionLevel.CLEAR if v < 0.15 else CongestionLevel.MODERATE if v < 0.40 else CongestionLevel.JAMMED
                for v in row
            ]
            for row in grid
        ]


class StationaryDetector:
    def __init__(self) -> None:
        self._cfg = CFG.stationary

    def update(self, track: TrackState, current_time: float) -> Optional[Incident]:
        speed = float(np.mean(list(track.speeds_kmh)[-5:])) if track.speeds_kmh else 0.0
        if speed < self._cfg.stationary_speed_epsilon:
            if track.last_moving_time == 0:
                track.last_moving_time = current_time
            elapsed = current_time - track.last_moving_time
            if elapsed >= self._cfg.stationary_threshold_sec:
                track.is_stationary = True
                return Incident(
                    incident_type=IncidentType.STATIONARY_VEHICLE,
                    track_id=track.track_id,
                    timestamp=current_time,
                    metadata={"duration_sec": round(elapsed, 2), "class": track.class_name},
                )
        else:
            track.last_moving_time = current_time
            track.is_stationary = False
        return None


class WrongWayDetector:
    def __init__(self) -> None:
        self._cfg = CFG.wrong_way

    def update(self, track: TrackState) -> Optional[Incident]:
        if len(track.trajectory) < self._cfg.min_track_length:
            return None
        pts = list(track.trajectory)[-self._cfg.min_track_length:]
        movement = (pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        if math.hypot(*movement) < 5:
            return None
        mv = self._normalize(movement)
        lane = (0.0, 1.0)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, mv[0] * lane[0] + mv[1] * lane[1]))))
        if angle > self._cfg.angle_tolerance_deg:
            return Incident(
                incident_type=IncidentType.WRONG_WAY,
                track_id=track.track_id,
                timestamp=0.0,
                metadata={"angle_deg": round(angle, 2), "class": track.class_name},
            )
        return None

    @staticmethod
    def _normalize(v: tuple[float, float]) -> tuple[float, float]:
        mag = math.hypot(v[0], v[1])
        return (v[0] / mag, v[1] / mag) if mag else (0.0, 0.0)


class CollisionDetector:
    def __init__(self) -> None:
        self._cfg = CFG.collision
        self._prev_iou: dict[tuple[int, int], float] = {}

    def update(self, tracks: dict[int, TrackState], current_time: float) -> list[Incident]:
        incidents: list[Incident] = []
        track_list = list(tracks.values())

        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                a, b = track_list[i], track_list[j]
                iou = self._compute_iou(a.bbox, b.bbox)
                key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                prev = self._prev_iou.get(key, 0.0)
                self._prev_iou[key] = iou
                if iou > self._cfg.iou_spike_threshold and (iou - prev) > self._cfg.iou_spike_delta:
                    incidents.append(Incident(
                        incident_type=IncidentType.COLLISION,
                        track_id=a.track_id,
                        timestamp=current_time,
                        metadata={"iou": round(iou, 4), "track_b": b.track_id},
                    ))

        for t in track_list:
            if self._detect_deceleration(t):
                incidents.append(Incident(
                    incident_type=IncidentType.COLLISION,
                    track_id=t.track_id,
                    timestamp=current_time,
                    metadata={"cause": "sudden_deceleration"},
                ))
        return incidents

    def _detect_deceleration(self, track: TrackState) -> bool:
        if len(track.speeds_kmh) < 6:
            return False
        speeds = list(track.speeds_kmh)
        recent = np.mean(speeds[-3:])
        older = np.mean(speeds[-6:-3])
        return (older - recent) > self._cfg.deceleration_threshold_kmh

    @staticmethod
    def _compute_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)


class AnalyticsEngine:
    def __init__(self) -> None:
        self.speed = SpeedEstimator()
        self.density = DensityAnalyzer()
        self.stationary = StationaryDetector()
        self.wrong_way = WrongWayDetector()
        self.collision = CollisionDetector()
        self.tracks: dict[int, TrackState] = {}
        self.incidents: list[Incident] = []

    def update_tracks(self, detections: list[tuple[int, int, float, float, float, float, float]], current_time: float) -> None:
        active_ids = set()
        for tid, cls_id, x1, y1, x2, y2, conf in detections:
            active_ids.add(tid)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            from app.detector import Detector
            class_name = Detector.class_name(cls_id)

            if tid not in self.tracks:
                self.tracks[tid] = TrackState(
                    track_id=tid, class_name=class_name,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(cx, cy), confidence=conf,
                    timestamp=current_time, first_seen=current_time,
                    last_moving_time=current_time,
                )
            t = self.tracks[tid]
            t.prev_bbox = t.bbox
            t.bbox = (int(x1), int(y1), int(x2), int(y2))
            t.center = (cx, cy)
            t.confidence = conf
            t.timestamp = current_time
            t.trajectory.append((cx, cy))

        stale = [tid for tid in self.tracks if tid not in active_ids]
        for tid in stale:
            del self.tracks[tid]

    def run_analytics(self, current_time: float, frame_shape: tuple[int, int] = (720, 1280)) -> list[Incident]:
        new_incidents: list[Incident] = []
        for t in self.tracks.values():
            spd = self.speed.update(t, current_time)
            if self.speed.is_overspeed(spd):
                new_incidents.append(Incident(
                    incident_type=IncidentType.OVERSPEED,
                    track_id=t.track_id,
                    timestamp=current_time,
                    metadata={"speed_kmh": round(spd, 1), "limit_kmh": CFG.speed.speed_limit_kmh},
                ))
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

    def get_density_map(self, frame_shape: tuple[int, int] = (720, 1280)) -> list[list[CongestionLevel]]:
        bboxes = [t.bbox for t in self.tracks.values()]
        return self.density.full_density_map(bboxes, frame_shape)

    def get_flow_stats(self) -> FlowStats:
        return FlowStats()
