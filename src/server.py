# src/server.py
"""
ATMS Web Server & Video Processing API.
Provides:
- Static file serving for web_visualizer, results, and uploads.
- Real-time video upload handling (MP4, AVI, MOV).
- Video transcoding (AVI -> MP4) for native browser playback.
- Execution of the full OpenCV + YOLO/Torchvision + CentroidTracker + CustomTrafficCNN
  + VisionToStateAdapter + RL Policy + Sequence-VAE pipeline on uploaded video files.
- Export of live telemetry to web_visualizer/rollout_data.js and results/hybrid_rollout.json.
"""

import os
import sys
import json
import math
import time
import threading
import traceback
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# Ensure project root in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.city_network import (
    CityNetworkEnv,
    INTERSECTIONS,
    FACILITIES,
    ROAD_SEGMENTS,
    CITY_WIDTH,
    CITY_HEIGHT,
)


# ---------------------------------------------------------------------------
# Global City Network Environment instance
# ---------------------------------------------------------------------------
_city_lock = threading.Lock()
_city_env = CityNetworkEnv()


# ---------------------------------------------------------------------------
# Global background-job state (protected by _job_lock)
# ---------------------------------------------------------------------------
_job_lock = threading.Lock()
_job = {
    "state": "idle",       # idle | preparing | analyzing | done | error
    "progress": 0,         # 0-100
    "result": None,        # pipeline payload when done
    "error": None,         # error message string
}


def _reset_job():
    """Reset JOB to idle (caller must hold _job_lock)."""
    _job["state"] = "idle"
    _job["progress"] = 0
    _job["result"] = None
    _job["error"] = None


def _progress_cb(pct):
    """Called from hybrid pipeline to update progress."""
    with _job_lock:
        _job["progress"] = int(pct)


# ---------------------------------------------------------------------------
# Video preparation: downscale, cap duration, resample fps
# ---------------------------------------------------------------------------
def prepare_video(src, dst=None, max_width=960, max_duration=20, target_fps=12):
    """
    Normalizes an uploaded video for fast inference:
    - Downscale so width <= max_width (keep aspect ratio, force even dims).
    - Cap duration at max_duration seconds.
    - Resample to ~target_fps by skipping frames.
    - Writes to dst (default uploads/prepared.mp4) using mp4v codec.
    Raises RuntimeError on failure.
    """
    import cv2
    if dst is None:
        dst = os.path.join(ROOT_DIR, "uploads", "prepared.mp4")

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {src}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        raise RuntimeError(f"Video reports 0 frames: {src}")

    # Compute frame skip to reach ~target_fps
    skip = max(1, round(src_fps / target_fps))
    out_fps = src_fps / skip

    # Compute max frames from duration cap
    max_out_frames = int(max_duration * out_fps)

    # Compute output dimensions
    if src_w > max_width:
        scale = max_width / src_w
        out_w = max_width
        out_h = int(src_h * scale)
    else:
        out_w = src_w
        out_h = src_h
    # Force even
    out_w = out_w if out_w % 2 == 0 else out_w + 1
    out_h = out_h if out_h % 2 == 0 else out_h + 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst, fourcc, out_fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {dst}")

    written = 0
    frame_idx = 0
    while written < max_out_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))
            out.write(frame)
            written += 1
        frame_idx += 1

    cap.release()
    out.release()

    if written == 0:
        raise RuntimeError(f"Prepared video has 0 frames from source: {src}")

    print(f"[Server] Prepared video: {src_w}x{src_h}@{src_fps:.0f}fps -> {out_w}x{out_h}@{out_fps:.1f}fps, {written} frames -> {dst}")
    return dst


def convert_avi_to_mp4(input_path, output_mp4_path=None):
    """
    Transcodes an AVI (or non-browser-native) video to standard MP4 using OpenCV.
    Ensures HTML5 video players can display the uploaded video natively.
    """
    import cv2
    if output_mp4_path is None:
        base, _ = os.path.splitext(input_path)
        output_mp4_path = f"{base}_web.mp4"

    if os.path.exists(output_mp4_path):
        return output_mp4_path

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return input_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Standard MP4 encoder
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"[Server] Converted {input_path} -> {output_mp4_path} ({w}x{h} @ {fps:.1f}fps)")
    return output_mp4_path


def _run_pipeline_job(save_path, playable_path, rel_url, max_steps):
    """Background thread: prepare video then run the hybrid pipeline."""
    try:
        from src.hybrid import run_hybrid_pipeline
        # 1. Prepare (downscale, cap duration, resample fps)

        with _job_lock:
            _job["state"] = "preparing"
            _job["progress"] = 5
        prepared = prepare_video(save_path, max_width=1920, target_fps=15)

        # 2. Analyze
        with _job_lock:
            _job["state"] = "analyzing"
            _job["progress"] = 20

        print(f"[Server] Running ATMS Hybrid Pipeline on prepared video: {prepared}...")
        payload = run_hybrid_pipeline(
            video_path=prepared,
            max_frames=max_steps,
            output_json=os.path.join(ROOT_DIR, "results", "hybrid_rollout.json"),
            output_js=os.path.join(ROOT_DIR, "web_visualizer", "rollout_data.js"),
            device="cpu",
            progress_callback=_progress_cb,
            stride_override=2,
            resize_width=1920,
            upload_mode=True,
        )

        # Point video_url at the original uploaded (playable) file
        payload["video_metadata"]["video_url"] = rel_url

        with _job_lock:
            _job["state"] = "done"
            _job["progress"] = 100
            _job["result"] = payload

        print(f"[Server] Background job finished. Delivered {len(payload['steps'])} steps.")

    except Exception as exc:
        traceback.print_exc()
        with _job_lock:
            _job["state"] = "error"
            _job["error"] = str(exc)
        print(f"[Server] Background job failed: {exc}")


class ATMSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler supporting CORS, Range requests, and Video Processing API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS for local cross-origin file:// and localhost requests
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length")
        super().end_headers()

    def _send_json_response(self, data, code=200):
        if isinstance(data, (bytes, bytearray)):
            payload = data
        else:
            payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/health", "/healthz", "/api/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            return
        if path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/web_visualizer/visualizer.html")
            self.end_headers()
            return
        super().do_HEAD()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Platform health probe endpoint (Render, Railway, Docker, Kubernetes)
        if path in ("/health", "/healthz"):
            self._send_json_response({"status": "ok", "service": "atms"})
            return

        # Root redirect
        if path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/web_visualizer/visualizer.html")
            self.end_headers()
            return

        # Status endpoint
        if path == "/api/status":
            with _city_lock:
                models_status = {
                    "custom_cnn": os.path.exists(os.path.join(ROOT_DIR, "models", "custom_cnn.pt")) and getattr(_city_env, "custom_cnn", None) is not None,
                    "sequence_gan": os.path.exists(os.path.join(ROOT_DIR, "models", "sequence_gan.pt")) and getattr(_city_env, "gan_generator", None) is not None,
                    "sequence_vae": os.path.exists(os.path.join(ROOT_DIR, "models", "sequence_vae.pt")) and getattr(_city_env, "vae_model", None) is not None,
                    "rl_policy": os.path.exists(os.path.join(ROOT_DIR, "models", "rl_policy.pt")) and getattr(_city_env, "policy_model", None) is not None,
                }
                city_summary = {
                    "intersections": len(INTERSECTIONS),
                    "active_incidents": len(_city_env.active_incidents),
                    "step": int(_city_env.step_idx),
                }
            resp = {
                "status": "ready",
                "message": "ATMS Server Operational",
                "models": models_status,
                "city_network": city_summary,
            }
            with _job_lock:
                resp["job"] = {
                    "state": _job["state"],
                    "progress": _job["progress"],
                    "error": _job["error"],
                }
                if _job["state"] == "done" and _job["result"] is not None:
                    resp["job"]["result"] = _job["result"]
            self._send_json_response(resp)
            return

        # City Network Topology endpoint
        if path == "/api/city/topology":
            topo = {
                "width": CITY_WIDTH,
                "height": CITY_HEIGHT,
                "intersections": INTERSECTIONS,
                "facilities": FACILITIES,
                "road_segments": ROAD_SEGMENTS,
            }
            self._send_json_response(topo)
            return

        # City Network State endpoint (Fully Thread-Safe)
        if path == "/api/city/state":
            with _city_lock:
                import numpy as np
                inter_states = _city_env.get_state()
                all_queues = [q for inter in _city_env.intersections.values() for q in inter.queues]
                avg_queue = float(round(float(np.mean(all_queues)), 2)) if all_queues else 0.0
                total_veh = float(round(float(sum(np.sum(inter.queues) for inter in _city_env.intersections.values())), 1))
                in_transit_veh = float(round(float(sum(_city_env.corridor_in_transit.values())), 1))
                vae_residuals = [inter.vae_residual for inter in _city_env.intersections.values()]
                avg_vae = float(round(float(np.mean(vae_residuals)), 2)) if vae_residuals else 0.0
                max_vae = float(round(float(np.max(vae_residuals)), 2)) if vae_residuals else 0.0

                state_data = {
                    "intersections": inter_states,
                    "incidents": list(_city_env.active_incidents),
                    "dispatches": list(_city_env.active_dispatches),
                    "step": int(_city_env.step_idx),
                    "preemption_active": bool(_city_env.preemption_active),
                    "avg_queue": avg_queue,
                    "total_vehicles": total_veh,
                    "in_transit_vehicles": in_transit_veh,
                    "system_total_vehicles": float(round(total_veh + in_transit_veh, 2)),
                    "vae_residual": avg_vae,
                    "max_vae_residual": max_vae,
                }
            self._send_json_response(state_data)
            return

        # Video status endpoint — returns whether a prepared video is available for playback
        if path == "/api/video/status":
            uploads_dir = os.path.join(ROOT_DIR, "uploads")
            prepared = os.path.join(uploads_dir, "prepared.mp4")
            candidates = []
            if os.path.isdir(uploads_dir):
                for f in os.listdir(uploads_dir):
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        fp = os.path.join(uploads_dir, f)
                        candidates.append((os.path.getmtime(fp), f))
            candidates.sort(reverse=True)
            if candidates:
                latest = candidates[0][1]
                self._send_json_response({
                    "available": True,
                    "url": f"/uploads/{latest}",
                    "prepared_url": "/uploads/prepared.mp4" if os.path.exists(prepared) else None,
                })
            else:
                self._send_json_response({"available": False, "url": None, "prepared_url": None})
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/api/process_video", "/api/upload_and_process"):
            self.handle_process_video(parsed)
        elif path == "/api/city/step":
            self.handle_city_step()
        elif path == "/api/city/incident":
            self.handle_city_incident()
        elif path == "/api/city/resolve_incident":
            self.handle_city_resolve_incident()
        elif path == "/api/city/reset":
            self.handle_city_reset()
        elif path == "/api/gan/generate":
            self.handle_gan_generate()
        elif path == "/api/vae/detect":
            self.handle_vae_detect()
        elif path == "/api/policy/predict":
            self.handle_policy_predict()
        else:
            self._send_json_response({"error": f"Unknown endpoint {path}"}, code=404)

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return {}
        if length > 0:
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def handle_city_step(self):
        body = self._read_json_body()
        actions = body.get("actions", None)
        if actions is not None and not isinstance(actions, dict):
            self._send_json_response({"error": "'actions' must be an object mapping intersection IDs to signal actions"}, code=400)
            return
        with _city_lock:
            try:
                state, reward, done, info = _city_env.step(actions)
            except (ValueError, TypeError) as exc:
                self._send_json_response({"error": str(exc)}, code=400)
                return
            payload = {
                "state": state,
                "reward": float(reward),
                "done": bool(done),
                "info": info
            }
        self._send_json_response(payload)

    def handle_city_incident(self):
        body = self._read_json_body()
        node_id = body.get("node_id", None)
        inc_type = body.get("type", "collision")
        if not isinstance(inc_type, str) or not inc_type.strip():
            inc_type = "collision"
        x = body.get("x", None)
        y = body.get("y", None)
        try:
            if x is not None:
                x = float(x)
                if not math.isfinite(x):
                    raise ValueError("x must be finite")
            if y is not None:
                y = float(y)
                if not math.isfinite(y):
                    raise ValueError("y must be finite")
        except (TypeError, ValueError):
            self._send_json_response({"error": "Incident coordinates x and y must be finite numbers"}, code=400)
            return
        approach = body.get("blocked_approach", None)
        with _city_lock:
            incident = _city_env.trigger_incident(
                location_node=node_id, incident_type=inc_type, x=x, y=y, blocked_approach=approach
            )
            payload = {"incident": incident, "status": "dispatched"}
        self._send_json_response(payload)

    def handle_city_resolve_incident(self):
        body = self._read_json_body()
        inc_id = body.get("id", None)
        with _city_lock:
            if inc_id:
                _city_env.resolve_incident(inc_id)
            else:
                _city_env.active_incidents.clear()
                _city_env.active_dispatches.clear()
                _city_env.preemption_active = False
                for inter in _city_env.intersections.values():
                    inter.preemption = False
            payload = {"status": "resolved"}
        self._send_json_response(payload)

    def handle_city_reset(self):
        with _city_lock:
            state = _city_env.reset()
            payload = {"state": state, "status": "reset"}
        self._send_json_response(payload)

    def handle_gan_generate(self):
        body = self._read_json_body()
        scenario = body.get("scenario", "rush_hour")
        if not isinstance(scenario, str) or not scenario.strip():
            scenario = "rush_hour"
        try:
            num_samples = int(body.get("num_samples", 1))
        except (TypeError, ValueError):
            self._send_json_response({"error": "num_samples must be an integer"}, code=400)
            return
        if num_samples < 1 or num_samples > 50:
            self._send_json_response({"error": "num_samples must be between 1 and 50"}, code=400)
            return
        with _city_lock:
            scenarios = _city_env.generate_traffic_scenarios(scenario_type=scenario, num_samples=num_samples)
            # Serialize numpy arrays if present
            formatted = []
            for sc in scenarios:
                seq = sc.get("sequence", [])
                if hasattr(seq, "tolist"):
                    seq = seq.tolist()
                formatted.append({
                    "sequence": seq,
                    "validated": sc.get("validated", False),
                    "status": sc.get("status", "generated")
                })
        self._send_json_response({"scenario": scenario, "results": formatted})

    def handle_vae_detect(self):
        body = self._read_json_body()
        seq = body.get("sequence", None)
        if not isinstance(seq, (list, tuple)) or len(seq) < 5:
            self._send_json_response({"error": "Sequence of length >= 5 required"}, code=400)
            return

        cleaned_seq = []
        for row in seq:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                self._send_json_response({"error": "Each sequence step must be a 4-element numeric vector"}, code=400)
                return
            try:
                vals = [float(v) for v in row]
            except (TypeError, ValueError):
                self._send_json_response({"error": "Sequence values must be numeric"}, code=400)
                return
            if any(isinstance(v, bool) for v in row) or not all(math.isfinite(v) for v in vals):
                self._send_json_response({"error": "Sequence values must be finite numbers"}, code=400)
                return
            cleaned_seq.append(vals)

        with _city_lock:
            if _city_env.vae_model is not None:
                from src.vae import detect_incident_anomaly
                import numpy as np
                seq_np = np.array(cleaned_seq, dtype=np.float32)
                res = detect_incident_anomaly(_city_env.vae_model, seq_np, _city_env.vae_th_warn, _city_env.vae_th_crit, device=_city_env.device)
            else:
                res = {"error": "Sequence VAE model not loaded"}
        self._send_json_response(res)

    def handle_policy_predict(self):
        body = self._read_json_body()
        st = body.get("state", None)
        if not isinstance(st, (list, tuple)) or len(st) != 7:
            self._send_json_response({"error": "7-dimensional state vector required"}, code=400)
            return
        try:
            if any(isinstance(v, bool) for v in st):
                raise ValueError("Boolean not allowed in state vector")
            st_floats = [float(v) for v in st]
            if not all(math.isfinite(v) for v in st_floats):
                raise ValueError("Non-finite value in state vector")
        except (TypeError, ValueError):
            self._send_json_response({"error": "7-dimensional numeric state vector required"}, code=400)
            return

        wait_timers = body.get("wait_timers", None)
        parsed_waits = None
        if wait_timers is not None:
            if not isinstance(wait_timers, (list, tuple)) or len(wait_timers) != 4:
                self._send_json_response({"error": "wait_timers must be a 4-element numeric list"}, code=400)
                return
            try:
                if any(isinstance(v, bool) for v in wait_timers):
                    raise ValueError("Boolean not allowed")
                parsed_waits = [float(v) for v in wait_timers]
                if not all(math.isfinite(v) for v in parsed_waits):
                    raise ValueError("Non-finite wait timer")
            except (TypeError, ValueError):
                self._send_json_response({"error": "wait_timers must be 4 finite numbers"}, code=400)
                return

        with _city_lock:
            if _city_env.policy_model is not None:
                import torch
                import numpy as np
                with torch.no_grad():
                    inp_t = torch.tensor([st_floats], dtype=torch.float32).to(_city_env.device)
                    out_p = _city_env.policy_model(inp_t)
                    probs = out_p.squeeze(0).cpu().numpy().tolist()
                    act = int(np.argmax(probs))
                resp = {
                    "action": act,
                    "probs": [float(round(p, 6)) for p in probs]
                }
                if parsed_waits is not None:
                    p_ns, p_ew = float(probs[0]), float(probs[1])
                    arm_scores = [
                        float(round(st_floats[idx] * (1.0 + 0.5 * (p_ns if idx < 2 else p_ew)) + 0.35 * parsed_waits[idx], 4))
                        for idx in range(4)
                    ]
                    resp["arm_scores"] = arm_scores
                    resp["recommended_arm"] = ("N", "S", "E", "W")[int(np.argmax(arm_scores))]
            else:
                resp = {"error": "Policy model not loaded"}
        self._send_json_response(resp)


    def handle_process_video(self, parsed):
        """Saves uploaded video and starts background processing job."""
        # Reject if a job is already running
        with _job_lock:
            if _job["state"] in ("preparing", "analyzing"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"started": False, "reason": "A job is already running"}).encode("utf-8"))
                return

        uploads_dir = os.path.join(ROOT_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        query_params = urllib.parse.parse_qs(parsed.query)
        filename = query_params.get("filename", [None])[0]
        if not filename:
            filename = self.headers.get("X-Filename", "traffic_input.mp4")
        filename = os.path.basename(filename)

        content_type = self.headers.get("Content-Type", "")
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            content_length = 0

        if content_length <= 0:
            self._send_json_response({"error": "Empty upload body"}, code=400)
            return

        save_path = os.path.join(uploads_dir, filename)

        print(f"[Server] Receiving video upload: {filename} ({content_length / 1024 / 1024:.2f} MB)...")

        # Parse multipart or raw stream
        if "multipart/form-data" in content_type:
            boundary = content_type.split("boundary=")[-1].strip()
            # Read all body
            body_bytes = self.rfile.read(content_length)
            # Find file part within boundary
            parts = body_bytes.split(f"--{boundary}".encode())
            file_data = None
            for p in parts:
                if b'filename=' in p and (b'\r\n\r\n' in p or b'\n\n' in p):
                    delimiter = b'\r\n\r\n' if b'\r\n\r\n' in p else b'\n\n'
                    header_part, content_part = p.split(delimiter, 1)
                    file_data = content_part.rstrip(b'\r\n')
                    break
            if file_data is not None:
                with open(save_path, "wb") as f:
                    f.write(file_data)
            else:
                with open(save_path, "wb") as f:
                    f.write(body_bytes)
        else:
            # Raw stream
            bytes_left = content_length
            chunk_size = 64 * 1024
            with open(save_path, "wb") as f:
                while bytes_left > 0:
                    read_amount = min(bytes_left, chunk_size)
                    chunk = self.rfile.read(read_amount)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_left -= len(chunk)

        print(f"[Server] Saved uploaded video to {save_path} ({os.path.getsize(save_path)} bytes)")

        # If file is an AVI or other format, transcode to MP4 for browser playback
        ext = os.path.splitext(filename)[1].lower()
        playable_path = save_path
        if ext in (".avi", ".mov", ".mkv"):
            try:
                playable_path = convert_avi_to_mp4(save_path)
            except Exception as e:
                print(f"[Server] Transcoding warning: {e}")
                playable_path = save_path

        try:
            max_steps = max(1, min(1000, int(query_params.get("max_steps", [100])[0])))
        except (TypeError, ValueError):
            max_steps = 100

        # Initialize JOB and start background thread
        with _job_lock:
            _reset_job()
            _job["state"] = "preparing"

        rel_url = f"/uploads/{os.path.basename(playable_path)}"

        t = threading.Thread(
            target=_run_pipeline_job,
            args=(save_path, playable_path, rel_url, max_steps),
            daemon=True,
        )
        t.start()

        self._send_json_response({"started": True})
        print(f"[Server] Upload accepted – background job started.")


_PERSISTENT_UPLOADS = {"13868599_3840_2160_24fps.mp4", "prepared.mp4"}


def _cleanup_old_uploads():
    """Delete unreferenced temporary files in uploads/ older than 24 hours."""
    uploads_dir = os.path.join(ROOT_DIR, "uploads")
    if not os.path.isdir(uploads_dir):
        return
    cutoff = time.time() - 86400
    for fname in os.listdir(uploads_dir):
        if fname in _PERSISTENT_UPLOADS or fname.startswith("."):
            continue
        fpath = os.path.join(uploads_dir, fname)
        try:
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                print(f"[Server] Cleaned up old upload: {fname}")
        except Exception:
            pass


def start_server(port=None, host=None):
    """Starts the multi-threaded ATMS web server."""
    if port is None:
        port = int(os.environ.get("PORT", 8000))
    if host is None:
        host = os.environ.get("HOST", "0.0.0.0")
    os.makedirs(os.path.join(ROOT_DIR, "uploads"), exist_ok=True)
    _cleanup_old_uploads()
    ThreadingHTTPServer.allow_reuse_address = True
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ATMSRequestHandler)
    print(f"\n=======================================================")
    print(f"[ATMS] Autonomous Traffic Monitoring & Web Server Live")
    print(f"[ATMS] Serving at: http://{host}:{port}/web_visualizer/visualizer.html")
    print(f"[ATMS] Root directory: {ROOT_DIR}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down ATMS server.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    default_port = int(os.environ.get("PORT", 8000))
    default_host = os.environ.get("HOST", "0.0.0.0")
    parser = argparse.ArgumentParser(description="ATMS Web Server & Video Processing API")
    parser.add_argument("--port", type=int, default=default_port, help=f"Server port (default: $PORT or {default_port})")
    parser.add_argument("--host", type=str, default=default_host, help=f"Bind address (default: $HOST or {default_host})")
    args = parser.parse_args()
    start_server(port=args.port, host=args.host)
