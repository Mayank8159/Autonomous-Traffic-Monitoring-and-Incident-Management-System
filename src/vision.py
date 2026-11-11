# src/vision.py
"""
Module 3: Video Perception, Vehicle Detection, Object Tracking, and ROI Density Analysis.
Supports real-world traffic video processing, configurable ROIs, persistent tracking IDs,
32x32 spatial occupancy grid generation, and custom CNN spatial feature extraction.
"""

import os
import math
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as TF
from src.tracker import CentroidTracker
from src.cnn_custom import load_custom_cnn


# ---------------------------------------------------------------------------
# Utility: Non-Maximum Suppression (IoU-based, no extra libraries)
# ---------------------------------------------------------------------------
def _nms_boxes(boxes, scores, iou_threshold=0.5):
    """Apply greedy NMS. boxes: list of [x1,y1,x2,y2], scores: list of float.
    Returns indices to keep."""
    if len(boxes) == 0:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class VideoTrafficMonitor:
    """
    Complete video perception pipeline:
    - Ingests video frames from file or webcam
    - Detects vehicles and pedestrians
    - Tracks objects with persistent IDs and trajectories
    - Evaluates lane-wise queues via configurable spatial ROIs
    - Constructs 32x32 spatial occupancy grids
    """
    def __init__(self, video_path=None, roi_config=None, device="cpu", use_custom_cnn=True):
        self.video_path = video_path
        self.device = device
        self.tracker = CentroidTracker(max_disappeared=15, max_distance=60.0)
        self.use_custom_cnn = use_custom_cnn

        # Load CustomTrafficCNN
        self.custom_cnn = load_custom_cnn(device=device) if use_custom_cnn else None

        # Setup vehicle detector
        self.detector = self._init_detector()

        # Default 4-lane ROIs for standard intersection video [x1, y1, x2, y2] normalized
        # North (top approach), South (bottom approach), East (right approach), West (left approach)
        self.roi_config = roi_config or {
            "North": [0.35, 0.05, 0.50, 0.40],
            "South": [0.50, 0.60, 0.65, 0.95],
            "East":  [0.60, 0.35, 0.95, 0.50],
            "West":  [0.05, 0.50, 0.40, 0.65],
        }

        # Background subtractor fallback for robust offline video analysis
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)

    def _init_detector(self):
        """Initializes torchvision object detector with safe CPU fallback."""
        try:
            # Attempt to load lightweight MobileNetV3 SSD detector
            weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
            model.to(self.device)
            model.eval()
            return model
        except Exception:
            # Fallback will use morphological motion and contour detection
            return None

    def detect_objects_frame(self, frame):
        """
        Detects vehicles (cars, buses, trucks, motorcycles) and pedestrians.
        Returns boxes [[x1, y1, x2, y2], ...], classes, and scores.
        """
        h, w = frame.shape[:2]
        boxes = []
        classes = []
        scores = []

        if self.detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_t = TF.to_tensor(rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    preds = self.detector(img_t)[0]

                # COCO classes: 1: person, 3: car, 4: motorcycle, 6: bus, 8: truck
                target_classes = {1: "pedestrian", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

                for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                    if score > 0.35 and label.item() in target_classes:
                        b = box.cpu().numpy()
                        boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        classes.append(target_classes[label.item()])
                        scores.append(float(score.item()))
            except Exception:
                pass

        # If deep detector yielded no boxes or is offline, use morphological motion detection
        if len(boxes) == 0:
            fg_mask = self.bg_subtractor.apply(frame)
            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 450:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bh / max(1, bw)
                    cls_name = "pedestrian" if aspect > 2.0 and area < 1200 else "car"
                    boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
                    classes.append(cls_name)
                    scores.append(0.78)

        return boxes, classes, scores

    def generate_occupancy_grid(self, active_tracks, grid_size=(32, 32), frame_shape=(480, 640)):
        """
        Constructs a 32x32 spatial occupancy grid (density matrix) from active tracked vehicles.
        Values range in [0.0, 1.0].
        """
        h, w = frame_shape[:2]
        grid = np.zeros(grid_size, dtype=np.float32)

        for track in active_tracks:
            cx, cy = track["centroid"]
            gx = int(np.clip((cx / w) * grid_size[1], 0, grid_size[1] - 1))
            gy = int(np.clip((cy / h) * grid_size[0], 0, grid_size[0] - 1))

            # Splat density around the centroid
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < grid_size[0] and 0 <= nx < grid_size[1]:
                        weight = 1.0 if (dx == 0 and dy == 0) else 0.5
                        grid[ny, nx] = min(1.0, grid[ny, nx] + weight)

        return grid

    def count_vehicles_in_rois(self, active_tracks, frame_shape=(480, 640)):
        """
        Counts tracked vehicles and pedestrians falling within the 4 approach ROIs:
        North, South, East, West.
        """
        h, w = frame_shape[:2]
        counts = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        class_dist = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "pedestrian": 0}

        for track in active_tracks:
            cx, cy = track["centroid"]
            cls_name = track["class"]
            class_dist[cls_name] = class_dist.get(cls_name, 0) + 1

            # Check which approach ROI contains the object
            assigned = False
            for lane_name, roi in self.roi_config.items():
                rx1, ry1, rx2, ry2 = roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    # Trucks and buses contribute more to queue weight than cars/pedestrians
                    weight = 1.5 if cls_name in ["truck", "bus"] else (0.5 if cls_name == "pedestrian" else 1.0)
                    counts[lane_name] += weight
                    assigned = True
                    break

            if not assigned:
                # Assign to nearest approach if inside junction perimeter
                if cy < h * 0.45:
                    counts["North"] += 0.5
                elif cy > h * 0.55:
                    counts["South"] += 0.5
                elif cx > w * 0.55:
                    counts["East"] += 0.5
                else:
                    counts["West"] += 0.5

        return counts, class_dist

    def process_frame(self, frame, fps=30.0):
        """
        Executes detection, tracking, ROI counting, occupancy grid, and custom CNN inference.
        """
        boxes, classes, scores = self.detect_objects_frame(frame)
        active_tracks = self.tracker.update(boxes, classes, scores=scores, fps=fps)

        # 4-approach counts
        counts_dict, class_dist = self.count_vehicles_in_rois(active_tracks, frame.shape)
        queue_vector = [counts_dict["North"], counts_dict["South"], counts_dict["East"], counts_dict["West"]]

        # 32x32 Occupancy grid
        occ_grid = self.generate_occupancy_grid(active_tracks, grid_size=(32, 32), frame_shape=frame.shape)

        # Custom CNN spatial estimation
        cnn_densities = queue_vector
        if self.custom_cnn is not None:
            # Resize frame to 64x64 for CustomTrafficCNN
            resized = cv2.resize(frame, (64, 64))
            tensor_in = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                cnn_densities = self.custom_cnn(tensor_in).squeeze(0).numpy().tolist()

        return {
            "tracks": active_tracks,
            "counts": counts_dict,
            "queue_vector": [float(round(q, 2)) for q in queue_vector],
            "cnn_densities": [float(round(q, 2)) for q in cnn_densities],
            "class_distribution": class_dist,
            "occupancy_grid": occ_grid.tolist(),
            "total_objects": len(active_tracks),
        }

    # ------------------------------------------------------------------
    # Upload-specific: tiled detection for small/distant vehicles
    # ------------------------------------------------------------------
    def detect_objects_tiled(self, frame, tile_cols=2, tile_rows=2,
                             overlap=0.15, score_threshold=0.30):
        """Run the existing detector on overlapping tiles, merge boxes back
        to full-frame coordinates, and NMS-deduplicate.  Falls back to
        single-frame detection if the deep detector is None."""
        if self.detector is None:
            return self.detect_objects_frame(frame)

        h, w = frame.shape[:2]
        all_boxes, all_classes, all_scores = [], [], []

        tile_w = int(w / tile_cols * (1 + overlap))
        tile_h = int(h / tile_rows * (1 + overlap))
        step_x = int(w / tile_cols)
        step_y = int(h / tile_rows)

        target_classes = {1: "pedestrian", 3: "car", 4: "motorcycle",
                          6: "bus", 8: "truck"}

        for r in range(tile_rows):
            for c in range(tile_cols):
                x0 = min(c * step_x, w - tile_w)
                y0 = min(r * step_y, h - tile_h)
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x0 + tile_w, w)
                y1 = min(y0 + tile_h, h)

                tile = frame[y0:y1, x0:x1]
                try:
                    rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                    img_t = TF.to_tensor(rgb).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        preds = self.detector(img_t)[0]
                    for box, sc, lab in zip(preds["boxes"], preds["scores"],
                                            preds["labels"]):
                        if sc > score_threshold and lab.item() in target_classes:
                            b = box.cpu().numpy()
                            all_boxes.append([float(b[0]) + x0,
                                              float(b[1]) + y0,
                                              float(b[2]) + x0,
                                              float(b[3]) + y0])
                            all_classes.append(target_classes[lab.item()])
                            all_scores.append(float(sc.item()))
                except Exception:
                    pass

        # Fallback: if tiling found nothing, try the whole-frame detector
        if len(all_boxes) == 0:
            return self.detect_objects_frame(frame)

        # NMS across tiles
        keep = _nms_boxes(all_boxes, all_scores, iou_threshold=0.5)
        boxes = [all_boxes[i] for i in keep]
        classes = [all_classes[i] for i in keep]
        scores = [all_scores[i] for i in keep]
        return boxes, classes, scores

    # ------------------------------------------------------------------
    # Upload-specific: classify tracks as moving vs parked
    # ------------------------------------------------------------------
    @staticmethod
    def classify_track_motion(track, frame_width, move_thresh_frac=0.015):
        """Return True if the track has moved more than move_thresh_frac
        of frame_width over its trajectory lifetime."""
        traj = track.get("trajectory", [])
        if len(traj) < 2:
            return False  # not enough history – assume stationary
        first = np.array(traj[0])
        last = np.array(traj[-1])
        displacement = float(np.linalg.norm(last - first))
        threshold = frame_width * move_thresh_frac
        return displacement > threshold

    # ------------------------------------------------------------------
    # Upload-specific: direction-based approach assignment
    # ------------------------------------------------------------------
    @staticmethod
    def assign_approach_by_direction(track, center_x, center_y, radius,
                                     frame_width, frame_height):
        """Assign a track to N/S/E/W based on its direction of travel
        relative to the intersection center.  Returns None if the track
        is outside the radius or has insufficient trajectory."""
        cx, cy = track["centroid"]
        dist = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        max_radius = radius
        if dist > max_radius:
            return None

        traj = track.get("trajectory", [])
        if len(traj) < 2:
            # Fall back to position-based quadrant
            dx = cx - center_x
            dy = cy - center_y
        else:
            # Use dominant direction of travel
            first = np.array(traj[0])
            last = np.array(traj[-1])
            delta = last - first
            dx, dy = float(delta[0]), float(delta[1])

        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            # No movement – use position relative to center
            dx = cx - center_x
            dy = cy - center_y

        # Dominant axis
        if abs(dy) >= abs(dx):
            return "North" if dy < 0 else "South"    # moving up = approaching from South toward North
        else:
            return "West" if dx < 0 else "East"

    # ------------------------------------------------------------------
    # Upload-specific: full frame processing with parked-car filtering
    # ------------------------------------------------------------------
    def process_frame_upload(self, frame, fps=30.0, intersection_center=None):
        """Like process_frame but uses tiled detection, filters parked
        vehicles, and assigns approaches by direction of travel.
        intersection_center: (cx, cy) in pixels or None for frame center.
        Returns dict with same keys as process_frame plus 'moving_tracks'
        and 'parked_tracks' lists."""
        h, w = frame.shape[:2]

        # Tiled detection
        boxes, classes, scores = self.detect_objects_tiled(frame)

        # Tracking
        active_tracks = self.tracker.update(boxes, classes, scores=scores, fps=fps)

        # Intersection center
        if intersection_center is None:
            ic_x, ic_y = w / 2.0, h / 2.0
        else:
            ic_x, ic_y = intersection_center

        # Radius: half the shorter dimension – covers the intersection area
        radius = min(w, h) * 0.45

        # Classify moving vs parked
        moving_tracks = []
        parked_tracks = []
        for tr in active_tracks:
            if self.classify_track_motion(tr, w):
                moving_tracks.append(tr)
            else:
                parked_tracks.append(tr)

        # Count only moving vehicles per approach (direction-based)
        counts = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        class_dist = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "pedestrian": 0}

        for tr in moving_tracks:
            cls_name = tr["class"]
            class_dist[cls_name] = class_dist.get(cls_name, 0) + 1
            approach = self.assign_approach_by_direction(
                tr, ic_x, ic_y, radius, w, h)
            if approach is not None:
                weight = 1.5 if cls_name in ["truck", "bus"] else (
                    0.5 if cls_name == "pedestrian" else 1.0)
                counts[approach] += weight

        queue_vector = [counts["North"], counts["South"],
                        counts["East"], counts["West"]]

        # Occupancy grid (from all active tracks for viz)
        occ_grid = self.generate_occupancy_grid(
            active_tracks, grid_size=(32, 32), frame_shape=frame.shape)

        # Custom CNN
        cnn_densities = queue_vector
        if self.custom_cnn is not None:
            resized = cv2.resize(frame, (64, 64))
            tensor_in = torch.from_numpy(
                resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                cnn_densities = self.custom_cnn(tensor_in).squeeze(0).numpy().tolist()

        return {
            "tracks": active_tracks,
            "moving_tracks": moving_tracks,
            "parked_tracks": parked_tracks,
            "counts": counts,
            "queue_vector": [float(round(q, 2)) for q in queue_vector],
            "cnn_densities": [float(round(q, 2)) for q in cnn_densities],
            "class_distribution": class_dist,
            "occupancy_grid": occ_grid.tolist(),
            "total_objects": len(active_tracks),
            "intersection_center": [float(ic_x), float(ic_y)],
        }


def generate_sample_traffic_video(output_path="sample_traffic.mp4", num_frames=150, fps=20):
    """
    Synthesizes a clean MP4 video with an intersection and moving vehicles.
    Ensures the user can immediately test video mode without needing external downloads.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Vehicle state simulation
    np.random.seed(101)
    vehicles = []
    # Northbound, Southbound, Eastbound, Westbound vehicles
    for i in range(12):
        direction = i % 4
        if direction == 0:  # North approach (moving down)
            vehicles.append({"x": w // 2 - 40, "y": -50 - i * 60, "vx": 0, "vy": 3.0, "color": (40, 40, 220), "cls": "car"})
        elif direction == 1:  # South approach (moving up)
            vehicles.append({"x": w // 2 + 30, "y": h + 50 + i * 60, "vx": 0, "vy": -2.8, "color": (220, 120, 30), "cls": "car"})
        elif direction == 2:  # East approach (moving left)
            vehicles.append({"x": w + 50 + i * 70, "y": h // 2 - 35, "vx": -3.2, "vy": 0, "color": (30, 210, 220), "cls": "bus"})
        else:  # West approach (moving right)
            vehicles.append({"x": -50 - i * 70, "y": h // 2 + 30, "vx": 3.0, "vy": 0, "color": (200, 40, 200), "cls": "truck"})

    for frame_idx in range(num_frames):
        # Background road surface
        frame = np.ones((h, w, 3), dtype=np.uint8) * 45
        # Grass
        frame[0:h // 2 - 60, 0:w // 2 - 60] = (40, 95, 45)
        frame[0:h // 2 - 60, w // 2 + 60:w] = (40, 95, 45)
        frame[h // 2 + 60:h, 0:w // 2 - 60] = (40, 95, 45)
        frame[h // 2 + 60:h, w // 2 + 60:w] = (40, 95, 45)

        # Road lanes (North-South vertical, East-West horizontal)
        cv2.rectangle(frame, (w // 2 - 60, 0), (w // 2 + 60, h), (70, 75, 80), -1)
        cv2.rectangle(frame, (0, h // 2 - 60), (w, h // 2 + 60), (70, 75, 80), -1)

        # Center lane dividers
        for y in range(0, h, 30):
            cv2.line(frame, (w // 2, y), (w // 2, y + 15), (255, 230, 50), 2)
        for x in range(0, w, 30):
            cv2.line(frame, (x, h // 2), (x + 15, h // 2), (255, 230, 50), 2)

        # Stop lines
        cv2.line(frame, (w // 2 - 60, h // 2 - 65), (w // 2, h // 2 - 65), (255, 255, 255), 3)
        cv2.line(frame, (w // 2, h // 2 + 65), (w // 2 + 60, h // 2 + 65), (255, 255, 255), 3)
        cv2.line(frame, (w // 2 + 65, h // 2 - 60), (w // 2 + 65, h // 2), (255, 255, 255), 3)
        cv2.line(frame, (w // 2 - 65, h // 2), (w // 2 - 65, h // 2 + 60), (255, 255, 255), 3)

        # Draw and update vehicles
        for v in vehicles:
            v["x"] += v["vx"]
            v["y"] += v["vy"]
            x, y = int(v["x"]), int(v["y"])
            bw, bh = (40, 20) if v["vx"] != 0 else (20, 35)
            if v["cls"] == "bus":
                bw, bh = (55, 24)
            elif v["cls"] == "truck":
                bw, bh = (50, 22)

            if -60 <= x <= w + 60 and -60 <= y <= h + 60:
                cv2.rectangle(frame, (x - bw // 2, y - bh // 2), (x + bw // 2, y + bh // 2), v["color"], -1)
                cv2.rectangle(frame, (x - bw // 2, y - bh // 2), (x + bw // 2, y + bh // 2), (255, 255, 255), 1)

        # Text banner
        cv2.putText(frame, f"ATMS Surveillance Feed | Frame {frame_idx + 1}/{num_frames}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Generated sample traffic video saved to {output_path}")
    return output_path
