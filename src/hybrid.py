# src/hybrid.py
"""
Hybrid AI-Based Traffic Monitoring & Autonomous Incident Management Pipeline.
Integrates:
- Real-World Video & Computer Vision (OpenCV + Detection + Tracking + Custom CNN)
- 7-D State Adapter
- Sequence-Based VAE Incident Detector (Normal / Warning / Critical)
- LSTM / GRU Temporal Trend Forecasting
- Deep RL & MLP Signal Decision Engine
- Closed-Loop 4-Way Traffic Signal Simulation
Exports comprehensive rollout frames for the unified dark-themed dashboard.
"""

import os
import cv2
import json
import numpy as np
import torch
import sys

from src.env import TrafficEnv
from src.vision import VideoTrafficMonitor, generate_sample_traffic_video
from src.adapter import VisionToStateAdapter
from src.vae import load_sequence_vae, detect_incident_anomaly
from src.temporal import analyze_traffic_trends
from src.gan import load_traffic_gan, generate_scenario
from src.model import MLPModel, RNNModel


SAFETY_DISCLAIMER = (
    "Real-world video inputs produce simulated AI signal recommendations for demonstration "
    "and evaluation only. This system does not physically control municipal traffic signals."
)


def _save_debug_frame(frame, vis_results, counts, step_idx, debug_dir):
    """Save an annotated debug image showing boxes and approach counts."""
    try:
        dbg = frame.copy()
        h, w = dbg.shape[:2]

        # Draw moving tracks in green, parked in gray
        for tr in vis_results.get("moving_tracks", []):
            b = tr["box"]
            cv2.rectangle(dbg, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 255, 0), 2)
            cv2.putText(dbg, f"#{tr['id']} {tr['class']}",
                        (int(b[0]), int(b[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for tr in vis_results.get("parked_tracks", []):
            b = tr["box"]
            cv2.rectangle(dbg, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (128, 128, 128), 1)

        # Draw intersection center and radius
        ic = vis_results.get("intersection_center", [w / 2, h / 2])
        cv2.circle(dbg, (int(ic[0]), int(ic[1])), 6, (255, 255, 0), -1)
        radius = int(min(w, h) * 0.45)
        cv2.circle(dbg, (int(ic[0]), int(ic[1])), radius, (255, 255, 0), 1)

        # Per-approach counts text
        y_txt = 25
        for approach in ["North", "South", "East", "West"]:
            val = counts.get(approach, 0.0)
            cv2.putText(dbg, f"{approach}: {val:.1f}", (10, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_txt += 20

        out_path = os.path.join(debug_dir, f"debug_step_{step_idx:04d}.jpg")
        cv2.imwrite(out_path, dbg)
    except Exception:
        pass


def run_hybrid_pipeline(video_path=None, max_frames=120, rl_policy=None,
                        use_custom_cnn=True, output_json="results/hybrid_rollout.json",
                        output_js="web_visualizer/rollout_data.js", device="cpu",
                        progress_callback=None, stride_override=None, resize_width=None,
                        upload_mode=False):
    """
    Executes the end-to-end hybrid system:
    Video -> Detection -> Tracking -> Adapter -> VAE -> Temporal -> RL Agent -> Simulation -> Dashboard Export
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("web_visualizer", exist_ok=True)

    # 1. Ensure video source
    if video_path is None or not os.path.exists(video_path):
        sample_path = "results/sample_traffic.mp4"
        if not os.path.exists(sample_path):
            generate_sample_traffic_video(sample_path, num_frames=max_frames)
        video_path = sample_path

    print(f"[Hybrid ATMS] Initializing video monitor on: {video_path}")
    monitor = VideoTrafficMonitor(video_path=video_path, device=device, use_custom_cnn=use_custom_cnn)

    # 2. State Adapter and Simulation Environment
    adapter = VisionToStateAdapter(alpha=0.40, max_queue=30.0)
    env = TrafficEnv(green_duration=5, red_duration=5, min_green=3, max_steps=max_frames)
    state = env.reset()

    # 3. Load VAE Anomaly Detector
    vae_model, th_warn, th_crit = load_sequence_vae("models/sequence_vae.pt", device=device)

    # 4. Load or Initialize RL Policy
    if rl_policy is None:
        rl_policy = MLPModel(input_dim=7, hidden_dims=[64, 32], dropout=0.0).to(device)
        # Check if saved weights exist
        if os.path.exists("models/rl_policy.pt"):
            try:
                rl_policy.load_state_dict(torch.load("models/rl_policy.pt", map_location=device))
            except Exception:
                pass
    rl_policy.eval()

    # 5. Load GAN for scenario generation analytics
    gan_model = load_traffic_gan("models/sequence_gan.pt", device=device)
    synthetic_rush_hour = generate_scenario(gan_model, scenario_type="rush_hour", num_samples=1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0:
        total_frames = max_frames

    stride = stride_override if stride_override is not None else (max(1, total_frames // max_frames) if total_frames > max_frames else 1)

    rollout_steps = []
    queue_history = []
    active_incidents = []

    # Upload-mode: actuated signal state (not used in demo path)
    _act_phase = 0          # 0 = NS green, 1 = EW green
    _act_phase_counter = 0  # steps in current phase
    _ACT_MIN_GREEN = 3

    # Upload-mode: debug output directory
    debug_dir = None
    if upload_mode:
        debug_dir = os.path.join(os.path.dirname(video_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)

    print(f"[Hybrid ATMS] Running inference on video: {total_frames} frames ({frame_w}x{frame_h} @ {fps:.1f}fps, stride={stride})...")

    step_idx = 0
    raw_frame_idx = 0
    effective_max = min(max_frames, (total_frames + stride - 1) // stride) if stride > 0 else max_frames
    while step_idx < max_frames and raw_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster inference if requested
        if resize_width and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            new_w = resize_width
            new_h = int(frame.shape[0] * scale)
            # Ensure even dimensions
            new_h = new_h if new_h % 2 == 0 else new_h + 1
            frame = cv2.resize(frame, (new_w, new_h))

        # Progress reporting
        if step_idx % 20 == 0 or step_idx == effective_max - 1:
            print(f"[Hybrid ATMS] frame {step_idx + 1}/{effective_max}")
        if progress_callback is not None:
            pct = int(20 + (step_idx / max(1, effective_max - 1)) * 80) if effective_max > 1 else 100
            progress_callback(pct)

        # A. Computer Vision Pipeline (Module 3)
        if upload_mode:
            vis_results = monitor.process_frame_upload(frame, fps=fps)
        else:
            vis_results = monitor.process_frame(frame, fps=fps)
        observed_queues = vis_results["queue_vector"]
        queue_history.append(observed_queues)
        if len(queue_history) > 10:
            queue_history.pop(0)

        # B. 7-D State Adapter (Seamless Interface)
        if upload_mode:
            # Use actuated phase derived from observed counts
            state_7d = adapter.counts_to_state(observed_queues, current_phase=_act_phase)
        else:
            state_7d = adapter.counts_to_state(observed_queues, current_phase=env.phase)

        # C. Incident & Anomaly Detection (Module 5B VAE)
        recent_seq = queue_history[-5:]
        if len(recent_seq) == 5:
            anomaly_info = detect_incident_anomaly(vae_model, recent_seq, th_warn, th_crit, device=device)
        else:
            anomaly_info = {
                "level": "NORMAL",
                "score": 1.25,
                "alert_message": "Traffic flow normal across all approaches",
                "affected_approach": "None",
                "approach_scores": [0.3, 0.3, 0.3, 0.3]
            }

        if anomaly_info["level"] != "NORMAL":
            active_incidents.append({
                "step": step_idx,
                "level": anomaly_info["level"],
                "message": anomaly_info["alert_message"],
                "approach": anomaly_info["affected_approach"]
            })

        # D. Temporal Traffic Forecasting (Module 4 LSTM / GRU)
        temporal_analysis = analyze_traffic_trends(queue_history)

        if upload_mode:
            # ---- Actuated signal logic for uploads ----
            ns_count = observed_queues[0] + observed_queues[1]
            ew_count = observed_queues[2] + observed_queues[3]
            desired_phase = 0 if ns_count >= ew_count else 1

            if _act_phase_counter == 0:
                _act_phase = desired_phase
                _act_phase_counter = 1
            elif desired_phase != _act_phase:
                if _act_phase_counter >= _ACT_MIN_GREEN:
                    _act_phase = desired_phase
                    _act_phase_counter = 1
                else:
                    _act_phase_counter += 1
            else:
                _act_phase_counter += 1

            recommended_action = _act_phase
            # Probabilities proportional to counts
            total_c = ns_count + ew_count + 1e-6
            probs_np = [float(ns_count / total_c), float(ew_count / total_c)]

            # Build env_info-compatible queues from observed counts
            env_queues = np.array(observed_queues, dtype=np.float32)
            avg_q = float(np.mean(env_queues))
            reward = -avg_q
            departed = 0.0  # no simulation departure in upload mode

            time_sec = float(round(raw_frame_idx / fps, 2)) if fps > 0 else float(step_idx)

            step_record = {
                "step": step_idx,
                "time_sec": time_sec,
                "video_frame_index": raw_frame_idx,
                "disclaimer": SAFETY_DISCLAIMER,
                "tracks": vis_results["tracks"][:25],
                "total_objects": vis_results["total_objects"],
                "class_distribution": vis_results["class_distribution"],
                "detected_counts": vis_results["counts"],
                "occupancy_grid": vis_results["occupancy_grid"],
                "cnn_densities": vis_results["cnn_densities"],
                "state_7d": [float(round(x, 2)) for x in state_7d],
                "action": recommended_action,
                "probs": [float(round(p, 4)) for p in probs_np],
                "phase": int(_act_phase),
                "phase_counter": int(_act_phase_counter),
                "queues": [float(round(q, 2)) for q in env_queues],
                "reward": float(round(reward, 2)),
                "departed": float(round(departed, 2)),
                "anomaly": anomaly_info,
                "temporal": temporal_analysis,
            }

            # Debug frame every ~10 steps
            if debug_dir and step_idx % 10 == 0:
                _save_debug_frame(frame, vis_results, vis_results["counts"],
                                  step_idx, debug_dir)

        else:
            # ---- Original demo path (RL policy + env.step) ----
            # E. Decision Engine: Advisory Signal Recommendation
            with torch.no_grad():
                st_tensor = torch.tensor(state_7d, dtype=torch.float32).unsqueeze(0).to(device)
                out = rl_policy(st_tensor)
                probs = out if torch.all(out >= 0) and torch.allclose(out.sum(dim=1), torch.ones(1), atol=1e-3) else torch.softmax(out, dim=1)
                probs_np = probs.squeeze().cpu().numpy().tolist()
                recommended_action = int(torch.argmax(probs, dim=1).item())

            # F. Closed-Loop 4-Way Simulation Step
            next_env_state, reward, done, env_info = env.step(recommended_action)

            time_sec = float(round(raw_frame_idx / fps, 2)) if fps > 0 else float(step_idx)

            step_record = {
                "step": step_idx,
                "time_sec": time_sec,
                "video_frame_index": raw_frame_idx,
                "disclaimer": SAFETY_DISCLAIMER,
                "tracks": vis_results["tracks"][:25],
                "total_objects": vis_results["total_objects"],
                "class_distribution": vis_results["class_distribution"],
                "detected_counts": vis_results["counts"],
                "occupancy_grid": vis_results["occupancy_grid"],
                "cnn_densities": vis_results["cnn_densities"],
                "state_7d": [float(round(x, 2)) for x in state_7d],
                "action": recommended_action,
                "probs": [float(round(p, 4)) for p in probs_np],
                "phase": int(env.phase),
                "phase_counter": int(env.phase_counter),
                "queues": [float(round(q, 2)) for q in env_info["queues"]],
                "reward": float(round(reward, 2)),
                "departed": float(round(env_info["departed"], 2)),
                "anomaly": anomaly_info,
                "temporal": temporal_analysis,
            }

        rollout_steps.append(step_record)
        step_idx += 1
        raw_frame_idx += stride

    cap.release()

    # Package complete dashboard payload
    rel_video_url = f"/uploads/{os.path.basename(video_path)}" if "uploads" in video_path.replace("\\", "/") else f"../{video_path.replace('\\', '/')}"
    dashboard_payload = {
        "disclaimer": SAFETY_DISCLAIMER,
        "video_metadata": {
            "file": os.path.basename(video_path),
            "video_url": rel_video_url,
            "fps": float(round(fps, 1)),
            "width": frame_w,
            "height": frame_h,
            "total_frames": total_frames,
            "processed_steps": len(rollout_steps),
            "duration_sec": float(round(total_frames / fps, 2)) if fps > 0 else 0.0,
        },
        "total_steps": len(rollout_steps),
        "steps": rollout_steps,
        "incident_history": active_incidents[-10:],
        "gan_analytics": synthetic_rush_hour,
        "summary": {
            "action_0_count": sum(1 for s in rollout_steps if s["action"] == 0),
            "action_1_count": sum(1 for s in rollout_steps if s["action"] == 1),
            "phase_0_count": sum(1 for s in rollout_steps if s["phase"] == 0),
            "phase_1_count": sum(1 for s in rollout_steps if s["phase"] == 1),
            "avg_queue": float(round(np.mean([np.mean(s["queues"]) for s in rollout_steps]), 2)) if rollout_steps else 0.0,
            "total_incidents": len(active_incidents),
        }
    }

    with open(output_json, "w") as f:
        json.dump(dashboard_payload, f, indent=2)

    with open(output_js, "w") as f:
        f.write("window.ATMS_ROLLOUT = ")
        json.dump(dashboard_payload, f)
        f.write(";\n")

    print(f"\n[Hybrid ATMS] Successfully completed {len(rollout_steps)} steps.")
    print(f"[Hybrid ATMS] Rollout exported to {output_json} and {output_js}")
    print(f"[Hybrid ATMS] Summary: Action 0: {dashboard_payload['summary']['action_0_count']}, Action 1: {dashboard_payload['summary']['action_1_count']}")
    print(f"[Hybrid ATMS] Signal Phase distribution: Phase 0: {dashboard_payload['summary']['phase_0_count']}, Phase 1: {dashboard_payload['summary']['phase_1_count']}")
    return dashboard_payload
