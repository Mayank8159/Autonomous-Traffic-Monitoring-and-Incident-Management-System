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

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Root redirect
        if path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/web_visualizer/visualizer.html")
            self.end_headers()
            return

        # Status endpoint
        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            models_status = {
                "custom_cnn": os.path.exists(os.path.join(ROOT_DIR, "models", "custom_cnn.pt")) and getattr(_city_env, "custom_cnn", None) is not None,
                "sequence_gan": os.path.exists(os.path.join(ROOT_DIR, "models", "sequence_gan.pt")) and getattr(_city_env, "gan_generator", None) is not None,
                "sequence_vae": os.path.exists(os.path.join(ROOT_DIR, "models", "sequence_vae.pt")) and getattr(_city_env, "vae_model", None) is not None,
                "rl_policy": os.path.exists(os.path.join(ROOT_DIR, "models", "rl_policy.pt")) and getattr(_city_env, "policy_model", None) is not None,
            }
            resp = {
                "status": "ready",
                "message": "ATMS Server Operational",
                "models": models_status,
                "city_network": {
                    "intersections": len(INTERSECTIONS),
                    "active_incidents": len(_city_env.active_incidents),
                    "step": _city_env.step_idx,
                }
            }
            with _job_lock:
                resp["job"] = {
                    "state": _job["state"],
                    "progress": _job["progress"],
                    "error": _job["error"],
                }
                if _job["state"] == "done" and _job["result"] is not None:
                    resp["job"]["result"] = _job["result"]
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        # City Network Topology endpoint
        if path == "/api/city/topology":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            topo = {
                "width": CITY_WIDTH,
                "height": CITY_HEIGHT,
                "intersections": INTERSECTIONS,
                "facilities": FACILITIES,
                "road_segments": ROAD_SEGMENTS,
            }
            self.wfile.write(json.dumps(topo).encode("utf-8"))
            return

        # City Network State endpoint (Fully Thread-Safe)
        if path == "/api/city/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
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
                serialized = json.dumps(state_data).encode("utf-8")
            self.wfile.write(serialized)
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
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Unknown endpoint {path}"}).encode("utf-8"))

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}
        return {}

    def handle_city_step(self):
        body = self._read_json_body()
        actions = body.get("actions", None)
        with _city_lock:
            state, reward, done, info = _city_env.step(actions)
            payload = json.dumps({
                "state": state,
                "reward": float(reward),
                "done": bool(done),
                "info": info
            }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def handle_city_incident(self):
        body = self._read_json_body()
        node_id = body.get("node_id", None)
        inc_type = body.get("type", "collision")
        x = body.get("x", None)
        y = body.get("y", None)
        approach = body.get("blocked_approach", None)
        with _city_lock:
            incident = _city_env.trigger_incident(
                location_node=node_id, incident_type=inc_type, x=x, y=y, blocked_approach=approach
            )
            payload = json.dumps({"incident": incident, "status": "dispatched"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

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
            payload = json.dumps({"status": "resolved"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def handle_city_reset(self):
        with _city_lock:
            state = _city_env.reset()
            payload = json.dumps({"state": state, "status": "reset"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def handle_gan_generate(self):
        body = self._read_json_body()
        scenario = body.get("scenario", "rush_hour")
        num_samples = int(body.get("num_samples", 1))
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
            payload = json.dumps({"scenario": scenario, "results": formatted}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def handle_vae_detect(self):
        body = self._read_json_body()
        seq = body.get("sequence", None)
        if not seq or len(seq) < 5:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Sequence of length >= 5 required"}).encode("utf-8"))
            return

        with _city_lock:
            if _city_env.vae_model is not None:
                from src.vae import detect_incident_anomaly
                import numpy as np
                seq_np = np.array(seq, dtype=np.float32)
                res = detect_incident_anomaly(_city_env.vae_model, seq_np, _city_env.vae_th_warn, _city_env.vae_th_crit, device=_city_env.device)
                payload = json.dumps(res).encode("utf-8")
            else:
                payload = json.dumps({"error": "Sequence VAE model not loaded"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def handle_policy_predict(self):
        body = self._read_json_body()
        st = body.get("state", None)
        if not st or len(st) != 7:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "7-dimensional state vector required"}).encode("utf-8"))
            return

        with _city_lock:
            if _city_env.policy_model is not None:
                import torch
                import numpy as np
                with torch.no_grad():
                    inp_t = torch.tensor([st], dtype=torch.float32).to(_city_env.device)
                    out_p = _city_env.policy_model(inp_t)
                    probs = out_p.squeeze(0).cpu().numpy().tolist()
                    act = int(np.argmax(probs))
                payload = json.dumps({
                    "action": act,
                    "probs": [float(round(p, 6)) for p in probs]
                }).encode("utf-8")
            else:
                payload = json.dumps({"error": "Policy model not loaded"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


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
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length <= 0:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Empty upload body"}).encode("utf-8"))
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

        max_steps = int(query_params.get("max_steps", [100])[0])

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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"started": True}).encode("utf-8"))
        print(f"[Server] Upload accepted – background job started.")


def _cleanup_old_uploads():
    """Delete files in uploads/ older than 1 hour. Never touches anything outside uploads/."""
    uploads_dir = os.path.join(ROOT_DIR, "uploads")
    if not os.path.isdir(uploads_dir):
        return
    cutoff = time.time() - 3600
    for fname in os.listdir(uploads_dir):
        fpath = os.path.join(uploads_dir, fname)
        try:
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                print(f"[Server] Cleaned up old upload: {fname}")
        except Exception:
            pass
    # Clean debug directory
    debug_dir = os.path.join(uploads_dir, "debug")
    if os.path.isdir(debug_dir):
        import shutil
        try:
            shutil.rmtree(debug_dir)
            print("[Server] Cleaned up uploads/debug/")
        except Exception:
            pass


def start_server(port=8000, host="0.0.0.0"):
    """Starts the multi-threaded ATMS web server."""
    os.makedirs(os.path.join(ROOT_DIR, "uploads"), exist_ok=True)
    _cleanup_old_uploads()
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ATMSRequestHandler)
    print(f"\n=======================================================")
    print(f"[ATMS] Autonomous Traffic Monitoring & Web Server Live")
    print(f"[ATMS] Serving at: http://localhost:{port}/web_visualizer/visualizer.html")
    print(f"[ATMS] Root directory: {ROOT_DIR}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down ATMS server.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATMS Web Server & Video Processing API")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()
    start_server(port=args.port, host=args.host)
