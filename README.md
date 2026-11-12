# Hybrid AI-Based Autonomous Traffic Monitoring & Incident Management System (ATMS)

<div align="center">

[![Test Suite](https://img.shields.io/badge/pytest-14%2F14%20PASSED-brightgreen.svg?style=for-the-badge&logo=pytest)](file:///Users/shreyan/Documents/Autonomous-Traffic-Monitoring-and-Incident-Management-System/tests)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED.svg?style=for-the-badge&logo=docker)](file:///Users/shreyan/Documents/Autonomous-Traffic-Monitoring-and-Incident-Management-System/Dockerfile)
[![Engine](https://img.shields.io/badge/Visualizer-2.5D%20Isometric%2060FPS-blueviolet.svg?style=for-the-badge)](file:///Users/shreyan/Documents/Autonomous-Traffic-Monitoring-and-Incident-Management-System/web_visualizer/visualizer.html)
[![License](https://img.shields.io/badge/license-Academic%20Citation-orange.svg?style=for-the-badge)](#-license--academic-citation)

**An end-to-end, multi-tier cyber-physical command center integrating real-time computer vision, deep generative neural modeling, reinforcement learning signal control, and a 2.5D isometric command visualizer.**

[Live Visualizer](http://localhost:8000/web_visualizer/visualizer.html) • [System Architecture](docs/ARCHITECTURE.md) • [Before vs. After Comparison](docs/BEFORE_VS_AFTER.md) • [Technical Walkthrough](docs/WALKTHROUGH.md) • [Deployment Guide](#-cloud-containerization--production-deployment)

</div>

---

> **⚠️ SAFETY & ETHICAL DISCLAIMER**  
> **Real-world video inputs produce simulated AI signal recommendations for demonstration and evaluation only. This system does not physically control municipal traffic signals or actuate road infrastructure.**

---

## 🚦 System Overview

The **Autonomous Traffic Monitoring and Incident Management System (ATMS)** is an end-to-end deep-learning urban traffic platform designed for high-density municipal grids. Upgraded from a foundational reinforcement learning simulation, ATMS unifies:

1. **Surveillance Computer Vision**: Real-world video stream ingestion, $2 \times 2$ tiled high-resolution inference with greedy IoU Non-Maximum Suppression (NMS), persistent Centroid Tracking with parked-vehicle motion filtering, directional approach ROIs, and a custom PyTorch spatial-density CNN.
2. **Deep Generative Modeling**: Conditional Sequence GAN (cGAN) synthesizing multi-regime queue trajectories through physical quality gates, and a Sequence $\beta$-VAE monitoring multi-approach queue dynamics for real-time anomaly alerting.
3. **Demand-Driven 4-Way Single-Signal AI Control**: Mathematical mutual exclusion ensuring at most **one** approach arm is Green or Amber at any intersection at any time ($\sum \mathbb{I} \le 1$), paired with **Standby All-Red** (0-second instant wake-up on approach arrival) and **Instant Gap-Out** ($\ge 1.0\text{s}$ switch when a green arm drains).
4. **Continuous Vehicle Kinematics**: Physics-based Intelligent Driver Model (IDM) with decoupled jam and physical stopped headways ($a \ge 0.035\text{ px/frame}^2$ fluid queue takeoff), smooth stop-line creeping ($d_{\text{stop}} \le 0.5\text{ px}$), $C^1$ Quadratic Bezier intersection turns, and $180^\circ$ Cubic Bezier turnaround loops.
5. **Incident Command & Dijkstra Preemption**: Dynamic emergency incident generation, Dijkstra shortest-path allocation of nearest police and ambulance units, siren shockwaves, and automated Green-Wave corridor preemption.
6. **2.5D Isometric Command Center**: Hardware-accelerated canvas engine with back-to-front depth sorting (Painter's algorithm), 3D extruded multi-tier vehicles with road-contact tires ($z=0$), Free Cam Orbit engine ($360^\circ$ yaw, $15^\circ-90^\circ$ pitch), Chase Cam follow mode, and a skeuomorphic telemetry inspector.

---

## 🏛️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Computer Vision & Ingestion Layer (Module 3)"]
        VIDEO["Real-World Video Feed / CCTV Stream<br>(MP4, AVI, MOV)"] --> PREP["Video Normalizer & Transcoder<br>(prepare_video: 1920px max, 15 FPS)"]
        PREP --> TILED["2x2 Tiled High-Res Detector<br>(Torchvision SSDLite / YOLOv8n)"]
        TILED --> NMS["Greedy IoU NMS (IoU >= 0.50)<br>Coordinate Remapping"]
        NMS --> TRACK["Centroid Tracker & Motion Filter<br>(Parked vs Moving Classification)"]
        TRACK --> ROIS["Directional Approach ROIs<br>(North, South, East, West)"]
        PREP --> OCC["32x32 Spatial Occupancy Grid"]
        OCC --> CNN["Custom PyTorch CNN<br>(3 Conv-BN-Pool Layers + FC Head)"]
        ROIS & CNN --> ADAPT["VisionToStateAdapter<br>(Exponential Moving Average α = 0.35)"]
    end

    subgraph MODELS["2. Deep Generative & Temporal Modeling (Modules 4 & 5)"]
        ADAPT --> S7D["Standard 7-D State Interface<br>[q_N, q_S, q_E, q_W, N_tot, p_NS, p_EW]"]
        S7D --> BUFFER["Temporal Buffer (History T=10)"]
        BUFFER --> VAE["Sequence β-VAE (GRU Encoder-Decoder)<br>Normal Reconstruction Modeling"]
        BUFFER --> FORECAST["Temporal Forecaster<br>(LSTM / GRU Multi-Step Horizon)"]
        GAN["Conditional Sequence GAN<br>(Multi-Condition Trajectory Generator)"] --> GATE["Physical Quality Gate<br>(Boundary & Rate-of-Change Checks)"]
        GATE -.->|Synthetic Scenarios| ADAPT
        VAE --> ANOMALY{"Reconstruction Error vs<br>Calibrated Thresholds"}
        ANOMALY -->|MSE > μ + 3.5σ| ALERT["🚨 Critical Anomaly Alert<br>(Incident Trigger)"]
        ANOMALY -->|Normal / Warning| LOG["Operations Telemetry Log"]
    end

    subgraph CONTROL["3. 4-Arm Demand-Driven AI Orchestrator (Modules 1 & 2)"]
        S7D --> RLPOL["PyTorch RL Policy (MLP 7->64->32->2)<br>(REINFORCE / DQN Policy Gradient)"]
        RLPOL --> SCORER["4-Arm Demand Scorer<br>(Per-Arm Queues + RL Prior + Anti-Starvation)"]
        SCORER --> FSM["Single-Signal State Machine<br>(Invariant: max 1 Green/Amber per junction)"]
        FSM --> MODES{"Traffic Demand Condition"}
        MODES -->|All 4 Arms Empty| STANDBY["Standby All-Red<br>(All Heads RED, 0.0s Instant Wake)"]
        MODES -->|Active Green Drained| GAPOUT["Instant Gap-Out<br>(Fast Switch after >= 1.0s)"]
        MODES -->|Regular Flow| CYCLE["Normal Holding Phase<br>(MinGreen 8.5-10s -> Amber 3.2s -> All-Red 1.8s)"]
    end

    subgraph KINEMATICS["4. Continuous Physics & Simulation Engine"]
        FSM --> HEADS["24 Signal Heads & Pavement Stop-Bars<br>(Independent Physical Display)"]
        HEADS --> IDM["Continuous Intelligent Driver Model (IDM)<br>(s0=15px, a_max=0.06, b_comf=0.09)"]
        IDM --> CREEP["Adaptive Stop-Bar Creep<br>(Smooth Pull-Up to 38px Stop Line)"]
        CREEP --> BEZIER["C1 Quadratic & Cubic Bezier Arcs<br>(90° Turns & 180° Perimeter Loops)"]
        BEZIER --> SEPARATION["2-Pass Fleet Collision Guards<br>(Same-Lane safeGap=28px, Cross minBumper=26.2px)"]
        ALERT --> DIJKSTRA["Dijkstra Emergency Dispatcher<br>(Nearest Police, Hospital, Fire Allocation)"]
        DIJKSTRA --> PREEMPT["Green-Wave Preemption Router<br>(Travel Corridor Signal Lock)"]
        PREEMPT --> FSM
    end

    subgraph UI["5. 2.5D Isometric Command Center"]
        SEPARATION & HEADS & DIJKSTRA --> PAINTER["Unified Depth-Sorted Render Queue<br>(Painter's Algorithm Back-to-Front)"]
        PAINTER --> RENDER["HTML5 2.5D Canvas Viewport<br>(3D Extruded Vehicles, Buildings, Shadows)"]
        RENDER --> ORBIT["Free Cam Orbit Engine<br>(360° Yaw, 15°-90° Pitch, 6 Presets)"]
        RENDER --> CHASE["Chase Cam / Follow Car HUD<br>(Real-Time Telemetry & Tracking)"]
        RENDER --> SIDEBAR["Skeuomorphic Inspector Panel<br>(4-Arm Queues, RL Confidence, VAE Gauge, Mini-Radar)"]
    end
```

---

## 📚 Deep Learning Curriculum & Theoretical Mapping

The system comprehensively fulfills all five core modules of the academic Deep Learning curriculum:

| Module | Core Concepts & Techniques | Implementation Files | CLI Verification Command |
| :--- | :--- | :--- | :--- |
| **Module 1: Foundations** | Multi-Layer Perceptron (MLP), non-linear activations (ReLU, Sigmoid, Softmax), Kaiming initialization, Cross-Entropy loss | [`src/model.py`](src/model.py), [`src/trainer.py`](src/trainer.py) | `python main.py --mode supervised --epochs 10` |
| **Module 2: Training & Optimization** | Backpropagation, Adam optimizer ($\beta_1=0.9, \beta_2=0.999$), BatchNorm1d, Dropout, Policy Gradient (REINFORCE), Deep Q-Learning (DQN) with target network synchronization | [`src/trainer.py`](src/trainer.py), [`src/model.py`](src/model.py), [`src/evaluate.py`](src/evaluate.py) | `python main.py --mode rl --algo reinforce`<br>`python main.py --mode rl --algo dqn` |
| **Module 3: Computer Vision & CNNs** | Real-world video stream ingestion, tiled object detection (SSDLite / YOLO), greedy IoU NMS, CentroidTracker, parked vs. moving motion filtering, directional approach ROIs, $32 \times 32$ spatial occupancy grid, custom PyTorch CNN (`Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AdaptiveAvgPool2d`), 7-D adapter with EMA | [`src/vision.py`](src/vision.py), [`src/tracker.py`](src/tracker.py), [`src/cnn_custom.py`](src/cnn_custom.py), [`src/adapter.py`](src/adapter.py) | `python main.py --mode cnn-train --epochs 5`<br>`python main.py --mode video --video-path results/sample_traffic.mp4` |
| **Module 4: Temporal Modeling** | Recurrent neural networks (LSTM & GRU), multi-step temporal queue forecasting, rate-of-change trend classification | [`src/temporal.py`](src/temporal.py) | `python main.py --mode hybrid --max-steps 100` |
| **Module 5A: Generative Models (GAN)** | Conditional Sequence GAN, GRU Generator, Bidirectional GRU Discriminator, BCEWithLogitsLoss, multi-regime queue synthesis, physical quality gate validation | [`src/gan.py`](src/gan.py) | `python main.py --mode gan --epochs 10 --scenario rush_hour` |
| **Module 5B: Generative Models (VAE)** | Sequence $\beta$-VAE, GRU encoder-decoder, Reparameterization trick ($z = \mu + \epsilon \odot \sigma$), KL Divergence, threshold calibration ($\mu + 2\sigma, \mu + 3.5\sigma$), real-time incident alerting | [`src/vae.py`](src/vae.py) | `python main.py --mode vae --epochs 8` |
| **City Network & Incident Command** | 6-intersection urban grid simulation, Dijkstra shortest-path emergency routing, automated police/ambulance dispatch, Green-Wave signal preemption, interactive 2.5D command center | [`src/city_network.py`](src/city_network.py), [`web_visualizer/visualizer.html`](web_visualizer/visualizer.html) | `python main.py --mode city --max-steps 100`<br>`python server.py --port 8000` |
| **Unified Integration** | Video-in-the-loop hybrid pipeline, non-blocking asynchronous REST server, automated health probes, and live surveillance console | [`src/hybrid.py`](src/hybrid.py), [`src/server.py`](src/server.py), [`server.py`](server.py) | `python main.py --mode hybrid --max-steps 100`<br>`python server.py` |

---

## 🚦 Demand-Driven 4-Way Single-Signal AI Controller

### Mutual Exclusion Invariant
Unlike legacy dual-phase controllers that set opposing approaches green simultaneously, ATMS governs each approach arm independently:

$$\text{activeArm} \in \{\text{North}, \text{South}, \text{East}, \text{West}\}$$

$$\sum_{a \in \{\text{N, S, E, W}\}} \mathbb{I}\left(\text{state}(a) \in \{\text{GREEN}, \text{AMBER}\}\right) \le 1 \quad \forall t, \forall j \in \{\text{INT\_1} \dots \text{INT\_6}\}$$

### Finite State Machine (FSM) Workflow

```mermaid
stateDiagram-v2
    [*] --> StandbyAllRed: All Approaches Empty (maxQ < 0.08)
    
    StandbyAllRed --> ActiveGreen: Vehicle Approaches Arm a* (0.0s Instant Wake)
    ActiveGreen --> ActiveAmber: Gap-Out (q_active < 0.08, elapsed >= 1.0s) OR Normal Cycle (elapsed >= minGreen)
    ActiveAmber --> AllRedClearance: Amber Timer Expires (3.2s)
    
    state AllRedClearance {
        [*] --> CheckingBoxOccupancy
        CheckingBoxOccupancy --> WaitingBoxClearance: Vehicle in Conflict Zone (33x33px)
        WaitingBoxClearance --> TargetArmSelected: Box Clear & Timer Expired (1.8s)
        CheckingBoxOccupancy --> TargetArmSelected: Box Clear & Timer Expired (1.8s)
    }
    
    AllRedClearance --> ActiveGreen: Target Arm a* Exists (Snap to GREEN)
    AllRedClearance --> StandbyAllRed: All Approaches Cleared (Return to Standby)
    
    ActiveGreen --> PreemptionGreen: Emergency Vehicle Preemption Active
    PreemptionGreen --> AllRedClearance: Preemption Corridor Cleared / Incident Resolved
```

### Demand Scoring & Prioritization Formula
For every approach arm $a \in \{\text{North}, \text{South}, \text{East}, \text{West}\}$:

$$\text{logit}(a) = \begin{cases}
-50.0 & \text{if } q(a) < 0.08 \text{ and } \max_{k} q(k) \ge 0.08 \\
0.6 \cdot p_{\text{axis}}(a) + 2.5 \cdot q(a) + \text{Starve}(a) + \text{Hold}(a) & \text{otherwise}
\end{cases}$$

Where:
- $p_{\text{axis}}(a)$: Axis guidance from PyTorch MLP forward pass ($p_{\text{NS}}$ for N/S, $p_{\text{EW}}$ for E/W).
- $q(a)$: Real-time vehicle demand computed dynamically from vehicles approaching within $325\text{px}$.
- $\text{Starve}(a) = \mathbb{I}(w_a \ge 25.0) \cdot [4.0 + 0.8(w_a - 25.0)]$ prevents queue starvation.
- $\text{Hold}(a) = \mathbb{I}(a = a_{\text{active}} \wedge q(a) \ge 0.2) \cdot 0.5$ stabilizes ongoing platoons.

---

## 🚗 Continuous Vehicle Kinematics & Physics Engine

```mermaid
flowchart LR
    subgraph KINEMATICS["Continuous Kinematics Pipeline"]
        IDM["IDM Longitudinal Accel<br>v_target = 0.85-1.25 px/frame<br>s0 = 15px, a_max = 0.06"] --> CREEP["Adaptive Stop-Bar Creep<br>s0_signal = 0.4px<br>Smooth Halt at stopLineDist <= 0.5px"]
        CREEP --> BEZIER["C1 Quadratic & Cubic Bezier Arcs<br>90° Turns & 180° Perimeter Loops"]
        BEZIER --> PASS7["Pass 7: Same-Lane Clamp<br>safeGap = 28.0px<br>Lead-Speed Velocity Capping"]
        PASS7 --> PASS8["Pass 8: 2-Pass Fleet Separation<br>Cross-Pair minBumper = 26.2px<br>turnProgress Position Relaxation"]
    end
```

1. **Fluid Queue Discharge**: Equilibrium jam parameter ($s_0 = 15\text{ px}$) is decoupled from physical stopped spacing ($28.0\text{ px}$). When the lead vehicle accelerates, trailing vehicles immediately produce positive acceleration ($a = 0.0428 \ge 0.035\text{ px/frame}^2$), eliminating phantom jams.
2. **Stop-Bar Pull-Up**: Full IDM formulation with $s_{0,\text{signal}} = 0.4\text{ px}$ allows vehicles to pull all the way to the $38\text{ px}$ stop line without freezing $25-35\text{ px}$ prematurely.
3. **Rigid Lane Adherence**: $0.000000\text{ px}$ lateral drift verified across all cruising and queuing vehicles.
4. **Non-Overlapping Collision Guards**: 2-pass fleet-wide longitudinal relaxation enforces $\ge 28.0\text{ px}$ same-lane bumper distance and $\ge 26.2\text{ px}$ cross-pair separation without snapping vehicles backward.

---

## 💻 2.5D Isometric Command Center & Interactive Controls

The front-end visualizer (`web_visualizer/visualizer.html`) provides a cybernetic command center with skeuomorphic and neumorphic elements:

### Viewport & Camera Engine
- **Free Cam Continuous Orbit**: Hold **Right Mouse Button** or **Shift + Left Mouse Button** and drag to rotate $360^\circ$ in yaw and $15^\circ - 90^\circ$ in pitch.
- **Pan & Zoom**: Left-click and drag to pan; mouse wheel to zoom ($0.35\times$ to $3.8\times$).
- **6 Instant Camera Presets**:
  - `SW Isometric (yaw=45°, pitch=30°)`: Default South-West city overview.
  - `SE Isometric (yaw=135°, pitch=30°)`: South-East avenue perspective.
  - `NE Isometric (yaw=225°, pitch=30°)`: North-East reverse perspective toward Downtown Core.
  - `NW Isometric (yaw=315°, pitch=30°)`: North-West arterial boulevard perspective.
  - `Bird's-Eye GIS (yaw=0°, pitch=90°)`: Orthographic municipal surveillance grid.
  - `Elevated Command (yaw=45°, pitch=25°)`: Low-angle dramatic street-level view.
- **Interactive Chase Cam**: Click any vehicle (or click the "Chase Cam / Follow Car" button) to lock the camera with smooth exponential lerp tracking. Displays a floating telemetry HUD with live velocity ($km/h$), kinematic state, corridor name, and upcoming intersection maneuver.

### Skeuomorphic Telemetry Inspector
- **Junction Telemetry**: Live approach queue gauges ($0 - 25.0$), independent 3-lens physical signal head indicators (North, South, East, West), AI model confidence bars, state tensor vectors, and sequence VAE anomaly meters.
- **Manual 1-Arm Actuation**: Override buttons to force single-arm green (`Force N`, `Force S`, `Force E`, `Force W`) or restore `AI Auto (4-Arm RL)`.
- **Emergency Command**: Fleet availability monitors (Police, EMS, Fire), active preemption corridors, and timestamped live dispatch event logs.
- **AI Model Core**: Architecture inspection showing PyTorch tensor shapes, parameters, and live forward pass inference latency ($< 0.05\text{ ms}$).

---

## 🌐 Cloud Containerization & Production Deployment

ATMS is production-ready and fully containerized with environment variable configuration and automated health monitoring:

### 1. Docker Deployment (Recommended)
```bash
# Build production Docker image
docker build -t atms-command-center .

# Run container on port 8000
docker run -d -p 8000:8000 -e PORT=8000 -e HOST=0.0.0.0 --name atms-app atms-command-center
```
Open in browser: `http://localhost:8000/` (automatically redirects to `/web_visualizer/visualizer.html`).

### 2. Cloud PaaS Deployment (Render / Railway / Fly.io / Heroku)
- **Render**: One-click deployment via `render.yaml` (`env: docker`, `healthCheckPath: /health`).
- **Railway**: Native container build via `railway.json` (`DOCKERFILE` builder, `/health` probe).
- **Procfile Platforms**: Configured via `Procfile` (`web: python server.py`), binding automatically to `$PORT`.

### 3. Native Python Production Server
```bash
PORT=8000 HOST=0.0.0.0 python server.py
```

---

## 🔌 REST API Reference

The server exposes a non-blocking, thread-safe HTTP REST API supporting CORS, HTTP Range requests for video streaming, and JSON endpoints:

| Method | Endpoint | Description | Request Payload | Response Sample |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` / `/healthz` | Container readiness & liveness probe | None | `{"status": "ok", "service": "atms"}` |
| `HEAD` | `/health` | Lightweight HTTP HEAD health check | None | Status `200 OK` (no body) |
| `GET` | `/api/status` | Server state, models loaded, active jobs | None | `{"status": "ready", "models": {...}, "city_network": {...}}` |
| `GET` | `/api/city/topology` | Urban network topology, nodes, corridors | None | `{"intersections": {...}, "facilities": {...}, "road_segments": [...]}` |
| `GET` | `/api/city/state` | Live 6-intersection state, queues, lights | None | `{"intersections": {...}, "avg_queue": 4.2, "vae_residual": 1.25}` |
| `POST` | `/api/city/step` | Advance network simulation by one step | `{"actions": {"INT_2": "N"}}` | `{"state": {...}, "reward": -3.8, "done": false}` |
| `POST` | `/api/city/incident` | Trigger emergency incident & dispatch | `{"location": "INT_2", "type": "collision"}` | `{"incident": {...}, "dispatch": {...}, "preemption_corridor": [...]}` |
| `POST` | `/api/city/resolve_incident` | Clear incident & release preemption | `{"incident_id": "INC-1"}` | `{"resolved": true, "active_incidents": 0}` |
| `POST` | `/api/policy/predict` | Evaluate PyTorch RL policy forward pass | `{"state": [3.2, 4.1, 2.0, 1.8, 11.1, 1.0, 0.0]}` | `{"action": 0, "probs": [0.72, 0.28], "arm_scores": {...}}` |
| `POST` | `/api/vae/detect` | Evaluate Sequence VAE reconstruction | `{"sequence": [[3.2, 4.1, 2.0, 1.8], ...]}` | `{"residual": 1.25, "is_incident": false, "level": "NORMAL"}` |
| `POST` | `/api/gan/generate` | Synthesize queue scenario via GAN | `{"scenario": "rush_hour", "num_samples": 3}` | `{"samples": [...], "validated": true}` |
| `POST` | `/api/process_video` | Ingest video file for CCTV pipeline | Multipart video file stream | `{"started": true}` |
| `GET` | `/api/video/status` | Check persistent CCTV video availability | None | `{"available": true, "url": "/uploads/prepared.mp4"}` |

---

## 🧪 Verification & Test Suite Matrix

The platform includes an automated test suite featuring **14 rigorous tests** with zero warnings or relaxed assertions:

```bash
# Execute entire test suite
python -m pytest tests/ -v --tb=short
```

```
============================= test session starts ==============================
rootdir: /Users/shreyan/Documents/Autonomous-Traffic-Monitoring-and-Incident-Management-System
collected 14 items

tests/test_api_endpoints.py::test_rest_api_endpoints PASSED              [  7%]
tests/test_ml_backend.py::test_models_exist_and_load PASSED              [ 14%]
tests/test_ml_backend.py::test_js_policy_weights_precision PASSED        [ 21%]
tests/test_ml_backend.py::test_city_network_zero_randomizers PASSED      [ 28%]
tests/test_ml_backend.py::test_city_network_corridor_conservation PASSED [ 35%]
tests/test_ml_backend.py::test_sequence_vae_anomaly_residual_spike PASSED [ 42%]
tests/test_ml_backend.py::test_custom_cnn_spatial_densities PASSED       [ 50%]
tests/test_ml_backend.py::test_sequence_gan_scenario_generation PASSED   [ 57%]
tests/test_ml_backend.py::test_city_network_single_signal_invariant_and_4arm_actuation PASSED [ 64%]
tests/test_ml_backend.py::test_backend_empty_arm_exclusion_and_wait_timers PASSED [ 71%]
tests/test_ml_backend.py::test_backend_standby_all_red_and_instant_wake PASSED [ 78%]
tests/test_ml_backend.py::test_backend_fast_gap_out_and_non_sequential_jump PASSED [ 85%]
tests/test_playwright_e2e.py::test_playwright_e2e PASSED                 [ 92%]
tests/test_vehicle_kinematics.py::test_vehicle_kinematics PASSED         [100%]

======================== 14 passed in 89.45s (0:01:29) =========================
```

---

## 🖼️ Visual Gallery

````carousel
![ATMS 2.5D City Command Center in Operation](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/city_command_screenshot.png)
<!-- slide -->
![Independent 4-Way Signal Heads & Illuminated Stop-Bars](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/independent_4way_signals.png)
<!-- slide -->
![True Queue Stacking at Red Signals (Zero Overlapping / 28px Bumper Gap)](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/queue_discipline_red_signal.png)
<!-- slide -->
![Fluid Queue Discharge: Sequential Platoon Takeoff on Green](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/fluid_queue_discharge.png)
<!-- slide -->
![Rigid Lane Discipline: Zero Lateral Drift (0.000000px)](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/lane_discipline_centerline.png)
<!-- slide -->
![3D Extruded Vehicles with Road-Contact Tires & Shadows](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/3d_vehicles_ground_contact.png)
<!-- slide -->
![Chase Cam Follow Mode with Telemetry HUD](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/chase_cam_telemetry.png)
<!-- slide -->
![Emergency Preemption Wave & Incident Scene](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/city_incident_screenshot.png)
<!-- slide -->
![CCTV Video Perception Mode with Multi-Vehicle Tracking](file:///Users/shreyan/.gemini/antigravity/brain/2be00328-83f1-4e49-8de4-16b28ce5974b/cctv_tab_screenshot.png)
````

---

## ⚖️ License & Academic Citation

Developed as an academic demonstration of hybrid artificial intelligence for autonomous traffic monitoring and incident management. All neural architectures adhere to syllabus requirements for Deep Learning.
