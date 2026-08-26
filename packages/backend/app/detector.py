"""YOLOv8 detection + ByteTrack multi-object tracking pipeline for Lambda."""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from boxmot import ByteTrack as BoXMOTByteTrack
except ImportError:
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ImportError:
        BYTETracker = None

from app.config import CFG, DetectionConfig, TrackerConfig, VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class _ByteTrackArgs:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.track_thresh = cfg.track_thresh
        self.match_thresh = cfg.match_thresh
        self.track_buffer = cfg.track_buffer
        self.min_hits = cfg.min_hits
        self.frame_rate = cfg.frame_rate


class Detector:
    """YOLOv8 + ByteTrack pipeline optimized for Lambda execution."""

    _instance: Optional["Detector"] = None
    _model: Optional[object] = None
    _tracker: Optional[object] = None

    def __new__(cls) -> "Detector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._model is not None:
            return
        self._det_cfg = CFG.detection
        self._trk_cfg = CFG.tracker
        self._load_model()

    def _load_model(self) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed")
        logger.info("Loading YOLOv8 model: %s", self._det_cfg.model_name)
        self._model = YOLO(self._det_cfg.model_name)
        self._init_tracker()
        logger.info("Detection pipeline warmed up.")

    def _init_tracker(self) -> None:
        try:
            self._tracker = BoXMOTByteTrack(
                track_thresh=self._trk_cfg.track_thresh,
                match_thresh=self._trk_cfg.match_thresh,
                track_buffer=self._trk_cfg.track_buffer,
                min_hits=self._trk_cfg.min_hits,
                frame_rate=self._trk_cfg.frame_rate,
            )
        except Exception:
            if BYTETracker is not None:
                self._tracker = BYTETracker(args=_ByteTrackArgs(self._trk_cfg))
            else:
                self._tracker = None

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, float, float, float, float, float]]:
        """Run detection + tracking. Returns [(track_id, class_id, x1, y1, x2, y2, conf), ...]"""
        results = self._model.predict(
            source=frame,
            conf=self._det_cfg.confidence_threshold,
            iou=self._det_cfg.iou_threshold,
            device=self._det_cfg.device,
            half=self._det_cfg.half_precision,
            max_det=self._det_cfg.max_detections,
            classes=list(self._det_cfg.classes),
            verbose=False,
        )

        detections = self._extract_detections(results)
        if detections.size == 0:
            return []

        if self._tracker is None:
            return self._fallback_ids(detections)

        try:
            output = self._tracker.update(detections, frame)
        except Exception:
            return self._fallback_ids(detections)

        if output is None or len(output) == 0:
            return []

        tracked = []
        for row in output:
            if len(row) < 7:
                continue
            x1, y1, x2, y2, tid, conf, cls_id = row[:7]
            tracked.append((int(tid), int(cls_id), float(x1), float(y1), float(x2), float(y2), float(conf)))
        return tracked

    @staticmethod
    def _extract_detections(results: object) -> np.ndarray:
        for r in results:
            if r.boxes is None or r.boxes.xyxy is None:
                return np.empty((0, 6), dtype=np.float32)
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()[:, None]
            cls = r.boxes.cls.cpu().numpy()[:, None]
            return np.hstack([xyxy, conf, cls])
        return np.empty((0, 6), dtype=np.float32)

    @staticmethod
    def _fallback_ids(detections: np.ndarray) -> list[tuple[int, int, float, float, float, float, float]]:
        return [
            (i, int(c), float(x1), float(y1), float(x2), float(y2), float(cf))
            for i, (x1, y1, x2, y2, cf, c) in enumerate(detections)
        ]

    @staticmethod
    def class_name(cls_id: int) -> str:
        return VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
