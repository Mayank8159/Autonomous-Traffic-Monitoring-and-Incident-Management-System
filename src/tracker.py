# src/tracker.py
"""
Object tracking module for ATMS.
Provides persistent object tracking (ID assignment, centroid history,
velocity estimation, and dwell time analysis) across consecutive video frames.
"""

import numpy as np
from collections import OrderedDict


class CentroidTracker:
    """
    Centroid and IOU-based multi-object tracker that maintains persistent
    track IDs, movement trajectories, dwell times, and velocity estimates.
    """
    def __init__(self, max_disappeared=15, max_distance=60.0):
        self.next_object_id = 0
        self.objects = OrderedDict()       # object_id -> (cx, cy)
        self.boxes = OrderedDict()         # object_id -> (x1, y1, x2, y2)
        self.classes = OrderedDict()       # object_id -> class_name
        self.disappeared = OrderedDict()   # object_id -> count of lost frames
        self.trajectories = OrderedDict()  # object_id -> list of (cx, cy)
        self.dwell_times = OrderedDict()   # object_id -> total frames tracked
        self.velocities = OrderedDict()    # object_id -> approximate px/sec
        self.scores = OrderedDict()        # object_id -> confidence score

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box, class_name="vehicle", score=0.85):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.boxes[obj_id] = box
        self.classes[obj_id] = class_name
        self.scores[obj_id] = float(score)
        self.disappeared[obj_id] = 0
        self.trajectories[obj_id] = [centroid]
        self.dwell_times[obj_id] = 1
        self.velocities[obj_id] = 0.0
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.boxes[obj_id]
        del self.classes[obj_id]
        if obj_id in self.scores:
            del self.scores[obj_id]
        del self.disappeared[obj_id]
        del self.trajectories[obj_id]
        del self.dwell_times[obj_id]
        del self.velocities[obj_id]

    def update(self, rects, class_names=None, scores=None, fps=30.0):
        """
        Updates tracked objects given bounding boxes in [x1, y1, x2, y2] format.
        Returns a list of active tracking dicts.
        """
        if class_names is None or len(class_names) != len(rects):
            class_names = ["vehicle"] * len(rects)
        if scores is None or len(scores) != len(rects):
            scores = [0.85] * len(rects)

        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.get_active_tracks()

        input_centroids = np.zeros((len(rects), 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i], class_names[i], scores[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Distance matrix between existing centroids and new input centroids
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                old_centroid = self.objects[obj_id]
                new_centroid = input_centroids[col]

                # Update velocity estimate (approximate pixels/sec)
                displacement = np.linalg.norm(new_centroid - old_centroid)
                velocity = float(displacement * fps)
                self.velocities[obj_id] = 0.7 * self.velocities[obj_id] + 0.3 * velocity

                self.objects[obj_id] = new_centroid
                self.boxes[obj_id] = rects[col]
                self.classes[obj_id] = class_names[col]
                self.scores[obj_id] = float(scores[col])
                self.disappeared[obj_id] = 0
                self.dwell_times[obj_id] += 1
                self.trajectories[obj_id].append(tuple(new_centroid))
                if len(self.trajectories[obj_id]) > 50:
                    self.trajectories[obj_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col], class_names[col], scores[col])

        return self.get_active_tracks()

    def get_active_tracks(self):
        """Return structured track objects."""
        active = []
        for obj_id in self.objects.keys():
            if self.disappeared[obj_id] == 0:
                active.append({
                    "id": int(obj_id),
                    "box": [float(b) for b in self.boxes[obj_id]],
                    "centroid": [float(c) for c in self.objects[obj_id]],
                    "class": self.classes[obj_id],
                    "score": float(round(self.scores.get(obj_id, 0.85), 2)),
                    "dwell_frames": int(self.dwell_times[obj_id]),
                    "velocity_px_s": float(round(self.velocities[obj_id], 1)),
                    "trajectory": [[float(pt[0]), float(pt[1])] for pt in self.trajectories[obj_id]],
                })
        return active
