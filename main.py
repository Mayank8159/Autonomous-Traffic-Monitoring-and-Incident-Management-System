"""
Main entry-point.

Runs the real-time stream processing loop:
  • Reads frames from an RTSP / file / webcam source.
  • Runs detection + tracking via ``Detector``.
  • Feeds detections into ``AnalyticsEngine``.
  • Dispatches alerts via ``AlertSystem``.
  • Renders all visual overlays with OpenCV.
  • Exposes a live-telemetry FastAPI server in a background thread.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import sys
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from alert_system import AlertSystem
from analytics import AnalyticsEngine, CongestionLevel, TrackState
from config import CFG, IncidentType, SystemConfig
from detector import Detector

logger = logging.getLogger("traffic_monitor")


# ---------------------------------------------------------------------------
# Thread-safe frame queue
# ---------------------------------------------------------------------------

class FrameQueue:
    """Bounded producer-consumer queue for non-blocking frame I/O."""

    def __init__(self, maxsize: int = 4) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)

    def put(self, frame: np.ndarray) -> None:
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(frame)

    def get(self) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=0.04)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Frame reader (separate thread)
# ---------------------------------------------------------------------------

class FrameReader(threading.Thread):
    """Reads frames from *source* and pushes them into a ``FrameQueue``."""

    def __init__(self, source: str, frame_q: FrameQueue,
                 stream_cfg: object, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self._source = source
        self._q = frame_q
        self._cfg = stream_cfg
        self.running = threading.Event()
        self.running.set()
        self.fps: float = 0.0

    def run(self) -> None:
        while self.running.is_set():
            cap = self._open_capture()
            if cap is None:
                time.sleep(self._cfg.reconnect_delay)  # type: ignore[union-attr]
                continue
            self._read_loop(cap)
            cap.release()
            logger.warning("Stream ended, reconnecting…")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        try:
            src = self._source
            if src.isdigit():
                src = int(src)
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.warning("Cannot open source: %s", self._source)
                return None
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cfg.buffer_size)  # type: ignore[union-attr]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)  # type: ignore[union-attr]
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)  # type: ignore[union-attr]
            logger.info("Stream opened: %s", self._source)
            return cap
        except Exception as exc:
            logger.error("Capture open error: %s", exc)
            return None

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        prev = time.perf_counter()
        while self.running.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            dt = now - prev
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            prev = now
            self._q.put(frame)

    def stop(self) -> None:
        self.running.clear()


# ---------------------------------------------------------------------------
# Visual overlay renderer
# ---------------------------------------------------------------------------

class Renderer:
    """Draws bounding boxes, trajectories, speed vectors, density heatmap,
    counting line, alerts, and HUD onto the frame."""

    def __init__(self, cfg: object) -> None:
        self._cfg = cfg

    @staticmethod
    def _color_for_class(cls_name: str) -> tuple:
        palette = {
            "car": (255, 180, 0),
            "truck": (0, 165, 255),
            "bus": (255, 0, 255),
            "motorcycle": (0, 255, 255),
            "bicycle": (0, 255, 0),
        }
        return palette.get(cls_name, (200, 200, 200))

    @staticmethod
    def _congestion_color(level: CongestionLevel) -> tuple:
        return {
            CongestionLevel.CLEAR: (0, 200, 0),
            CongestionLevel.MODERATE: (0, 200, 255),
            CongestionLevel.JAMMED: (0, 0, 255),
        }[level]

    def draw_all(self, frame: np.ndarray, tracks: dict,
                 analytics: AnalyticsEngine, fps: float,
                 frame_count: int, recent_incidents: list) -> np.ndarray:
        vis = frame.copy()
        self._draw_density_heatmap(vis, analytics)
        self._draw_counting_line(vis)
        self._draw_tracks(vis, tracks)
        self._draw_hud(vis, fps, frame_count, analytics)
        self._draw_incident_banners(vis, recent_incidents)
        return vis

    def _draw_density_heatmap(self, frame: np.ndarray, analytics: AnalyticsEngine) -> None:
        if not self._cfg.show_density_heatmap:  # type: ignore[union-attr]
            return
        grid = analytics.get_density_map()
        if not grid:
            return
        h, w = frame.shape[:2]
        rows, cols = len(grid), len(grid[0])
        cell_h, cell_w = h // rows, w // cols
        overlay = frame.copy()
        for r in range(rows):
            for c in range(cols):
                color = self._congestion_color(grid[r][c])
                x1, y1 = c * cell_w, r * cell_h
                cv2.rectangle(overlay, (x1, y1), (x1 + cell_w, y1 + cell_h), color, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    def _draw_counting_line(self, frame: np.ndarray) -> None:
        if not self._cfg.show_counting_line:  # type: ignore[union-attr]
            return
        from config import CFG as _C
        p1, p2 = _C.roi.counting_line
        cv2.line(frame, p1, p2, (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def _draw_tracks(self, frame: np.ndarray, tracks: dict) -> None:
        cfg = self._cfg
        for tid, t in tracks.items():
            color = self._color_for_class(t.class_name)
            x1, y1, x2, y2 = t.bbox
            if cfg.show_bounding_boxes:  # type: ignore[union-attr]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.bbox_thickness)  # type: ignore[union-attr]
                label = f"ID:{tid} {t.class_name} {t.confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, color, cfg.font_thickness)  # type: ignore[union-attr]

            if cfg.show_trajectories and len(t.trajectory) > 1:  # type: ignore[union-attr]
                pts = list(t.trajectory)[-cfg.trajectory_length:]  # type: ignore[union-attr]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thick = max(1, int(2 * alpha))
                    cv2.line(frame, pts[i - 1], pts[i], color, thick)

            if cfg.show_speed_vectors and len(t.trajectory) >= 2:  # type: ignore[union-attr]
                tail = t.trajectory[-1]
                head = t.trajectory[-min(8, len(t.trajectory))]
                cv2.arrowedLine(frame, head, tail, (255, 255, 255), 2, tipLength=0.3)

            if t.is_stationary:
                cv2.putText(frame, "STATIONARY", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            speeds = list(t.speeds_kmh)
            if speeds:
                spd = speeds[-1]
                cv2.putText(frame, f"{spd:.1f} km/h", (x1, y2 + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_hud(self, frame: np.ndarray, fps: float,
                  frame_count: int, analytics: AnalyticsEngine) -> None:
        h, w = frame.shape[:2]
        stats = analytics.get_flow_stats()
        texts = [
            f"FPS: {fps:.1f}",
            f"Frame: {frame_count}",
            f"Active Tracks: {len(analytics.tracks)}",
            f"Entry: {stats.entry_count}  Exit: {stats.exit_count}",
            f"Flow Rate: {stats.flow_rate_per_min:.1f}/min",
        ]
        # background
        cv2.rectangle(frame, (10, 10), (340, 20 + 28 * len(texts)), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (340, 20 + 28 * len(texts)), (255, 255, 255), 1)
        for i, txt in enumerate(texts):
            cv2.putText(frame, txt, (18, 38 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_incident_banners(self, frame: np.ndarray,
                               incidents: list) -> None:
        if not self._cfg.show_alerts or not incidents:  # type: ignore[union-attr]
            return
        h, frame_w = frame.shape[:2]
        y = h - 20
        for inc in incidents[-5:]:
            txt = f"[ALERT] {inc['type'].upper()} — track {inc['track_id']}"
            cv2.rectangle(frame, (5, y - 22), (frame_w - 5, y + 5), (0, 0, 200), -1)
            cv2.putText(frame, txt, (12, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y -= 32


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

class TrafficMonitor:
    """Orchestrates reading, detection, analytics, alerts, and rendering."""

    def __init__(self, cfg: Optional[SystemConfig] = None,
                 source: Optional[str] = None) -> None:
        self._cfg = cfg or CFG
        self._source = source or self._cfg.stream.source
        self._frame_q = FrameQueue(maxsize=4)
        self._reader = FrameReader(self._source, self._frame_q, self._cfg.stream)
        self._detector = Detector()
        self._analytics: Optional[AnalyticsEngine] = None
        self._alert = AlertSystem()
        self._renderer = Renderer(self._cfg.visualization)
        self._loop = asyncio.new_event_loop()
        self._running = threading.Event()

    def _init_analytics(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        fps = self._reader.fps or self._cfg.tracker.frame_rate
        self._analytics = AnalyticsEngine(w, h, fps)
        # Bind analytics engine to the FastAPI telemetry server
        import api as _api
        _api.bind_analytics(self._analytics, self._alert.recent_alerts)

    def run(self) -> None:
        """Blocking call — runs until 'q' is pressed or stream ends."""
        self._running.set()
        self._reader.start()

        # Start the API server in a background daemon thread
        import api as _api
        _api.start_api_server(
            host=self._cfg.api.host,
            port=self._cfg.api.port,
        )

        logger.info("Traffic Monitor started. Press 'q' to quit.")

        try:
            while self._running.is_set():
                frame = self._frame_q.get()
                if frame is None:
                    time.sleep(0.005)
                    continue
                if self._analytics is None:
                    self._init_analytics(frame)
                self._process_frame(frame)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _process_frame(self, frame: np.ndarray) -> None:
        t0 = time.perf_counter()
        detections = self._detector.detect(frame)
        now = time.time()

        if self._analytics is None:
            return
        self._analytics.update_tracks(detections, now)
        new_incidents = self._analytics.run_analytics(now)

        # async alert dispatch (non-blocking)
        if new_incidents:
            asyncio.run_coroutine_threadsafe(
                self._alert.process_incidents(new_incidents, frame),
                self._loop,
            )

        fps = self._reader.fps
        vis = self._renderer.draw_all(
            frame, self._analytics.tracks, self._analytics,
            fps, self._analytics.frame_count,
            self._alert.recent_alerts,
        )

        cv2.imshow("Autonomous Traffic Monitor", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._running.clear()

    def _shutdown(self) -> None:
        logger.info("Shutting down…")
        self._reader.stop()
        self._reader.join(timeout=3)
        cv2.destroyAllWindows()
        try:
            self._loop.run_until_complete(self._alert.shutdown())
        except RuntimeError:
            pass
        if self._loop.is_running():
            self._loop.stop()
        logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous Traffic Monitoring System")
    p.add_argument("--source", "-s", default=CFG.stream.source,
                   help="Video source: RTSP URL, file path, or camera index")
    p.add_argument("--model", "-m", default=CFG.detection.model_name,
                   help="YOLOv8 model name/path")
    p.add_argument("--conf", type=float, default=CFG.detection.confidence_threshold,
                   help="Detection confidence threshold")
    p.add_argument("--device", default=CFG.detection.device,
                   help="Inference device (cpu / 0 / 0,1)")
    p.add_argument("--speed-limit", type=float, default=CFG.speed.speed_limit_kmh,
                   help="Speed limit in km/h")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless without OpenCV window")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Override config from CLI
    CFG.stream.source = args.source  # type: ignore[attr-defined]
    CFG.detection.model_name = args.model  # type: ignore[attr-defined]
    CFG.detection.confidence_threshold = args.conf  # type: ignore[attr-defined]
    CFG.detection.device = args.device  # type: ignore[attr-defined]
    CFG.speed.speed_limit_kmh = args.speed_limit  # type: ignore[attr-defined]

    if args.no_display:
        from config import VisualizationConfig
        CFG.visualization = VisualizationConfig(
            show_bounding_boxes=False, show_trajectories=False,
            show_speed_vectors=False, show_density_heatmap=False,
            show_counting_line=False, show_alerts=False,
        )

    monitor = TrafficMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
