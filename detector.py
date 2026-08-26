"""
Detection & tracking pipeline.

Wraps Ultralytics YOLOv8 inference and ByteTrack multi-object tracking into
a single synchronous ``Detector.detect(frame)`` call that returns enriched
detections with persistent track IDs.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

try:
    from boxmot import BotSort, DeepOcSORT, StrongSORT, ByteTrack as BoXMOTByteTrack
except ImportError:
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ImportError:
        BYTETracker = None  # type: ignore[assignment,misc]

from config import CFG, DetectionConfig, TrackerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ByteTrack config shim (ultralytics native tracker fallback)
# ---------------------------------------------------------------------------

class _ByteTrackArgs:
    """Minimal namespace expected by ultralytics BYTETracker."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.track_thresh = cfg.track_thresh
        self.match_thresh = cfg.match_thresh
        self.track_buffer = cfg.track_buffer
        self.min_hits = cfg.min_hits
        self.frame_rate = cfg.frame_rate


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """End-to-end YOLOv8 + ByteTrack pipeline.

    Usage::

        det = Detector()
        frame = cv2.imread("test.jpg")
        detections = det.detect(frame)
        # detections: [(track_id, class_id, x1, y1, x2, y2, conf), ...]
    """

    def __init__(self, det_cfg: Optional[DetectionConfig] = None,
                 trk_cfg: Optional[TrackerConfig] = None) -> None:
        self._det_cfg = det_cfg or CFG.detection
        self._trk_cfg = trk_cfg or CFG.tracker
        self._model: Optional[object] = None
        self._tracker: Optional[object] = None
        self._warm = False

    # ---- lazy init ---------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._warm:
            return
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed — run `pip install ultralytics`"
            )
        logger.info("Loading YOLOv8 model: %s (device=%s)", self._det_cfg.model_name, self._det_cfg.device)
        self._model = YOLO(self._det_cfg.model_name)
        self._init_tracker()
        self._warm = True
        logger.info("Detection + tracking pipeline warmed up.")

    def _init_tracker(self) -> None:
        """Attempt boxmot ByteTrack first, then fall back to ultralytics native."""
        try:
            self._tracker = BoXMOTByteTrack(
                track_thresh=self._trk_cfg.track_thresh,
                match_thresh=self._trk_cfg.match_thresh,
                track_buffer=self._trk_cfg.track_buffer,
                min_hits=self._trk_cfg.min_hits,
                frame_rate=self._trk_cfg.frame_rate,
            )
            logger.info("Using boxmot ByteTrack tracker.")
        except Exception:
            if BYTETracker is not None:
                self._tracker = BYTETracker(args=_ByteTrackArgs(self._trk_cfg))
                logger.info("Using ultralytics native BYTETracker.")
            else:
                logger.warning("No tracker available — running without tracking.")
                self._tracker = None

    # ---- public API --------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, float, float, float, int]]:
        """Run detection + tracking on *frame*.

        Returns a list of tuples::

            [(track_id, class_id, x1, y1, x2, y2, confidence), ...]

        Only vehicle classes listed in ``CFG.detection.classes`` are kept.
        """
        self._ensure_model()

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

        detections_np = self._extract_detections(results)
        if detections_np.size == 0:
            return []

        tracked = self._track(detections_np, frame)
        return tracked

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _extract_detections(results: object) -> np.ndarray:
        """Pull raw xyxy+conf+cls array from ultralytics Results."""
        for r in results:  # type: ignore[union-attr]
            if r.boxes is None or r.boxes.xyxy is None:
                return np.empty((0, 6), dtype=np.float32)
            xyxy = r.boxes.xyxy.cpu().numpy()          # (N, 4)
            conf = r.boxes.conf.cpu().numpy()[:, None]  # (N, 1)
            cls = r.boxes.cls.cpu().numpy()[:, None]    # (N, 1)
            return np.hstack([xyxy, conf, cls])         # (N, 6)
        return np.empty((0, 6), dtype=np.float32)

    def _track(self, detections: np.ndarray,
               frame: np.ndarray) -> List[Tuple[int, int, float, float, float, int]]:
        """Feed detections into the tracker and return ID-enriched list."""
        if self._tracker is None:
            return self._fallback_no_tracker(detections)

        try:
            if hasattr(self._tracker, "update"):
                output = self._tracker.update(detections, frame)
            else:
                return self._fallback_no_tracker(detections)
        except Exception as exc:
            logger.warning("Tracker failed (%s) — falling back.", exc)
            return self._fallback_no_tracker(detections)

        if output is None or len(output) == 0:
            return []

        results: List[Tuple[int, int, float, float, float, int]] = []
        for row in output:
            if len(row) < 7:
                continue
            x1, y1, x2, y2, tid, conf, cls_id = row[:7]
            results.append((
                int(tid),
                int(cls_id),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(conf),
            ))
        return results

    @staticmethod
    def _fallback_no_tracker(detections: np.ndarray) -> List[Tuple[int, int, float, float, float, int]]:
        """Synthesize monotonically increasing IDs when no tracker is available."""
        results: List[Tuple[int, int, float, float, float, int]] = []
        for i, row in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = row
            results.append((i, int(cls_id), float(x1), float(y1), float(x2), float(y2), float(conf)))
        return results
