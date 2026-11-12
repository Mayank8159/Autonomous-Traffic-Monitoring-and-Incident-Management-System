# Hybrid AI-Based Autonomous Traffic Monitoring & Incident Management System (ATMS)

> **⚠️ SAFETY & ETHICAL DISCLAIMER**  
> **Real-world video inputs produce simulated AI signal recommendations for demonstration and evaluation only. This system does not physically control municipal traffic signals or actuate road infrastructure.**

---

## 🚦 Project Overview

The **Autonomous Traffic Monitoring and Incident Management System (ATMS)** is an end-to-end, hybrid deep-learning platform designed for modern urban intersections. Upgraded from a foundational reinforcement learning simulation, ATMS integrates real-world computer vision surveillance, deep generative sequence modeling, temporal forecasting, and adaptive reinforcement learning control into a unified, modular architecture.

The project encompasses all five core modules of the academic Deep Learning curriculum:
1. **Module 1:** Neural network fundamentals (Perceptron, MLP, non-linear activations, Softmax policy).
2. **Module 2:** Deep learning training dynamics (Backprop, Kaiming He initialization, Adam, BatchNorm, Dropout, MSE/Cross-Entropy, REINFORCE, DQN).
3. **Module 3:** Convolutional Neural Networks (Real-world video processing, tiled SSDLite/YOLO vehicle detection, multi-object tracking with parked-car filtering, directional approach assignment, custom PyTorch CNN for 4-approach spatial density estimation, and 7-D state adaptation).
4. **Module 4:** Recurrent sequence modeling (LSTM & GRU for queue forecasting and trend classification).
5. **Module 5:** Deep Generative Models (Conditional Sequence GAN for queue synthesis and Sequence $\beta$-VAE for real-time anomaly and incident detection).

---

## 📚 Deep Learning Syllabus Mapping

| Module | Core Concepts & Techniques | Implementation Files | CLI Verification Command |
| :--- | :--- | :--- | :--- |
| **Module 1: Foundations** | Perceptron, Multi-Layer Perceptron (MLP), Sigmoid, ReLU, Softmax output | [`src/model.py`](src/model.py), [`src/trainer.py`](src/trainer.py) | `python main.py --mode supervised` |
| **Module 2: Training & Optimization** | Backpropagation, Kaiming He Init, Adam optimizer, BatchNorm1d, Dropout, MSE/BCE Loss, Policy Gradient (REINFORCE), Deep Q-Learning (DQN) with target network | [`src/trainer.py`](src/trainer.py), [`src/model.py`](src/model.py), [`src/evaluate.py`](src/evaluate.py) | `python main.py --mode rl --algo reinforce`<br>`python main.py --mode rl --algo dqn` |
| **Module 3: Computer Vision & CNNs** | Real-world video stream ingestion, Object detection (Torchvision SSDLite/YOLO), 2×2 tiled high-resolution inference with IoU NMS, Centroid tracking, parked vs. moving motion classification, directional approach ROIs, 32×32 occupancy grid, Custom PyTorch CNN (`Conv2d`, `BatchNorm2d`, `MaxPool2d`, `ReLU`, `Linear`), 7-D adapter | [`src/vision.py`](src/vision.py), [`src/tracker.py`](src/tracker.py), [`src/cnn_custom.py`](src/cnn_custom.py), [`src/adapter.py`](src/adapter.py) | `python main.py --mode cnn-train --epochs 5`<br>`python main.py --mode video --video-path results/sample_traffic.mp4` |
| **Module 4: Temporal Modeling** | Recurrent neural networks (LSTM & GRU), multi-step temporal queue forecasting, rate-of-change trend classification | [`src/temporal.py`](src/temporal.py) | Integrated in hybrid pipeline (`--mode hybrid`) |
| **Module 5A: Generative Adversarial Networks** | Conditional Sequence GAN, Generator (GRU + Linear + Sigmoid scaling), Discriminator (BiGRU + Linear), multi-condition queue synthesis, physical quality gate validation | [`src/gan.py`](src/gan.py) | `python main.py --mode gan --epochs 10 --scenario rush_hour` |
| **Deep Generative Models (VAE)** | Sequence $\beta$-VAE, GRU encoder-decoder, Reparameterization trick, Gaussian latent space, KL Divergence, threshold calibration, anomaly alerting | [`src/vae.py`](src/vae.py) | `python main.py --mode vae --epochs 8` |
| **City Network & Incident Command** | 6-intersection urban network simulation, Dijkstra emergency routing, automated police & ambulance dispatch, AI green-wave preemption, and interactive 2D GIS visualizer | [`src/city_network.py`](src/city_network.py), [`web_visualizer/visualizer.html`](web_visualizer/visualizer.html) | `python main.py --mode city --max-steps 100`<br>`python server.py --port 8000` |
| **Unified Integration** | End-to-end hybrid video-in-the-loop pipeline, non-blocking asynchronous REST server, and surveillance visualizer | [`src/hybrid.py`](src/hybrid.py), [`src/server.py`](src/server.py), [`server.py`](server.py), [`web_visualizer/visualizer.html`](web_visualizer/visualizer.html) | `python main.py --mode hybrid --max-steps 100`<br>`python main.py --mode server --port 8000` |


---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph S1["Module 3: Computer Vision Pipeline"]
        V[Real-World Traffic Video / Camera Feed] --> VD[SSDLite / YOLO Detector<br>2x2 Overlap Tiles & IoU NMS]
        VD --> TRK[Centroid Tracker & Motion Filter<br>Moving vs. Parked Vehicles]
        TRK --> ROI[Directional Approach ROIs<br>North, South, East, West]
        V --> GRID[32x32 Spatial Occupancy Grid]
        GRID --> CNN[Custom PyTorch CNN<br>Conv2d + BatchNorm2d + MaxPool2d]
        ROI --> ADAPT[VisionToStateAdapter<br>EMA Smoothing alpha=0.35]
        CNN --> ADAPT
    end

    subgraph S2["Module 5: Deep Generative Models"]
        GAN[Conditional Sequence GAN<br>Generator & Discriminator] -->|Synthetic Scenarios| VAL[Physical Validation Gate<br>Bounds, Continuity, Plausibility]
        VAL --> ADAPT
        HIST[(Temporal History Buffer)] --> VAE[Sequence Beta-VAE<br>Normal Traffic Reconstruction]
        VAE --> ADET{Reconstruction Error<br>Threshold Calibration}
        ADET -->|Score > Critical| ALARM["🚨 Critical Traffic Anomaly —<br>Investigation Required"]
        ADET -->|Normal / Warning| MON[Routine Log Monitor]
    end

    subgraph S3["Module 4 & RL Control"]
        ADAPT --> S7D["Standard 7-D State Interface<br>[q_N, q_S, q_E, q_W, N_tot, p_NS, p_EW]"]
        S7D --> GRU[Temporal Predictor<br>LSTM / GRU Forecasting]
        S7D --> RL[RL Policy / Actuated Engine<br>min_green=3 Constraint]
        RL --> ACT[Phase Recommendation<br>Action 0: NS Green | Action 1: EW Green]
    end

    subgraph S4["Surveillance Dashboard"]
        ACT --> DASH[web_visualizer/visualizer.html]
        V --> DASH
        ALARM --> DASH
        GRU --> DASH
    end
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- Git
- Modern Web Browser (Chrome, Edge, Firefox, or Safari)

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/Mayank8159/Autonomous-Traffic-Monitoring-and-Incident-Management-System.git
cd Autonomous-Traffic-Monitoring-and-Incident-Management-System

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide & CLI Commands

### 1. Launch ATMS Web Server & Real-World Video Upload Console (Recommended)
Starts the local ATMS server supporting direct in-browser video upload (MP4, AVI, MOV), automatic video transcoding, live OpenCV frame ingestion, and Deep Learning inference:
```bash
python main.py --mode server --port 8000
# Or directly:
python server.py --port 8000
```
*Open in browser:* `http://localhost:8000/web_visualizer/visualizer.html`  
*Features:* Drag-and-drop or upload any real traffic video from your computer, view real vehicle bounding boxes with confidence scores and persistent tracking IDs, observe real-time 32×32 occupancy grid heatmaps, and watch the 4-way intersection adapt to the AI signal recommendation!

- **Non-Blocking Background Processing:** The upload handler (`POST /api/process_video` or `POST /api/upload_and_process`) saves the raw video, starts a daemon background job thread, and immediately returns `{"started": true}`. The frontend visualizer polls `GET /api/status` every 1.5s to display live job state (`idle`, `preparing`, `analyzing`, `done`, `error`) and progress (0–100%).
- **Concurrency Guard:** Only one background job runs at a time; if an analysis is in progress, duplicate uploads are safely rejected with `{"started": false}`. The server uses Python's `ThreadingHTTPServer` so status polling and telemetry requests remain fully responsive during inference.
- **Video Preprocessing & Normalization:** Uploaded videos are normalized prior to inference by `prepare_video()`: downscaled to $\le 1920$ px width (preserving aspect ratio and even dimensions), duration capped at 20 seconds, resampled to $\sim 15$ fps via frame skipping, and written to `uploads/prepared.mp4` using the `mp4v` codec.
- **Actuated Signal Decisions & Debug Previews:** Real-world uploads trigger actuated signal control with minimum green durations (`min_green=3`) and export annotated debug verification frames every 10 steps to `uploads/debug/` (green boxes for moving vehicles, gray for parked, with intersection radius and counts drawn).
- **Startup Cleanup:** On server launch, old temporary files in `uploads/` older than 1 hour and the `uploads/debug/` folder are automatically purged.

### 2. Hybrid Pipeline (CLI Mode)
Executes the unified vision-in-the-loop pipeline from the command line: ingests surveillance video, runs detector + tracker + custom CNN, adapts to 7-D state, performs VAE incident monitoring, forecasts queues with GRU, evaluates the RL policy, and exports real-time dashboard telemetry.
```bash
python main.py --mode hybrid --video-path results/sample_traffic.mp4 --max-steps 100
```
*Generated output:* `results/hybrid_rollout.json` and `web_visualizer/rollout_data.js`.  
*CLI options:* Supports `stride_override` (e.g., stride 2 for uploads), `resize_width` (downscales frames exceeding width), `progress_callback` (mapped to 20–100% progress), and periodic console telemetry (`[Hybrid ATMS] frame i/total` every 20 frames).

### 2. Train Custom PyTorch CNN (Module 3)
Trains the 3-layer `CustomTrafficCNN` on 64×64 spatial occupancy grids to estimate multi-approach vehicle density:
```bash
python main.py --mode cnn-train --epochs 5 --lr 0.001
```
*Saved checkpoint:* `models/custom_cnn.pt`

### 3. Video Processing & Computer Vision Inspection (Module 3)
Processes a video file (or auto-generates a synthetic surveillance test video if not present), displays real-time detection, tracking IDs, ROI counts, and outputs trajectory analytics:
```bash
python main.py --mode video --video-path results/sample_traffic.mp4 --max-steps 100
```

### 4. Train & Validate Conditional Sequence GAN (Module 5A)
Trains the Sequence GAN to synthesize multi-step traffic queue trajectories across 6 operational conditions (`normal`, `moderate`, `heavy`, `rush_hour`, `uneven`, `surge`), validating generated trajectories against physical plausibility gates:
```bash
python main.py --mode gan --epochs 10 --scenario rush_hour
```
*Saved checkpoint:* `models/sequence_gan.pt`

### 5. Train Sequence $\beta$-VAE & Calibrate Anomaly Thresholds (Module 5B)
Trains the Sequence VAE on normal traffic dynamics, computes reconstruction statistics, establishes $\mu + 2\sigma$ (warning) and $\mu + 3.5\sigma$ (critical) thresholds, and validates incident detection:
```bash
python main.py --mode vae --epochs 8
```
*Saved checkpoint:* `models/sequence_vae.pt`

### 6. Reinforcement Learning Signal Control (Module 2)
Trains and evaluates RL traffic control policies with anti-starvation rewards and minimum green constraints:
```bash
# Policy Gradient (REINFORCE)
python main.py --mode rl --algo reinforce --epochs 30 --max-steps 100

# Deep Q-Network (DQN) with Target Network Synchronization
python main.py --mode rl --algo dqn --epochs 30 --max-steps 100
```
*Saved checkpoint:* `models/rl_policy.pt`

### 7. Supervised MLP Baseline (Module 1)
Trains and evaluates the foundational Multi-Layer Perceptron using Backpropagation and Adam:
```bash
python main.py --mode supervised --epochs 10 --batch-size 32
```

---

## 🔬 Technical Implementation Details

### Module 3: Computer Vision & Spatial CNN
- **Detection & Tracking:** Uses Torchvision `ssdlite320_mobilenet_v3_large` (with morphological color-fallback) to detect vehicles. Detections feed into `CentroidTracker` which computes centroid displacement, bounding box IOU, persistent track IDs, instant velocity ($px/\Delta t$), and dwell time.
- **Tiled High-Resolution Inference:** In addition to standard single-frame inference, `detect_objects_tiled()` splits high-resolution and aerial frames into $2 \times 2$ overlapping tiles with 15% overlap and a 0.30 score threshold. Bounding boxes are re-mapped to full-frame coordinates and merged using greedy IoU Non-Maximum Suppression (`_nms_boxes`, IoU threshold 0.5) to capture small, top-down vehicles.
- **Motion Classification & Parked-Car Filtering:** `classify_track_motion()` evaluates cumulative track displacement over trajectory history. A vehicle is classified as moving only if its net displacement exceeds 1.5% of the frame width (`move_thresh_frac=0.015`). Stationary vehicles parked along road edges or lots are excluded from approach queues.
- **Directional Approach ROIs:** `assign_approach_by_direction()` assigns moving vehicles to North, South, East, or West based on their dominant trajectory vector ($\Delta x$ vs. $\Delta y$) toward the intersection center within a radial perimeter ($r = 0.45 \times \min(w, h)$), filtering out extraneous background parking lots.
- **Custom PyTorch CNN Architecture:**
  - `Conv2d(1, 16, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(16)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
  - `Conv2d(16, 32, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
  - `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool2d((4, 4))`
  - Fully Connected Head: `Linear(64*4*4, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(128, 4)` predicting density for North, South, East, West.
- **7-D State Adapter:** Fuses detector counts and CNN spatial density estimates into $[q_{\text{North}}, q_{\text{South}}, q_{\text{East}}, q_{\text{West}}, N_{\text{total}}, \phi_{\text{NS}}, \phi_{\text{EW}}]$ with Exponential Moving Average (EMA, $\alpha=0.35$).

### Module 5A: Conditional Sequence GAN
- **Generator:** Takes latent noise $z \sim \mathcal{N}(0, I_{16})$ and one-hot condition vector $c \in \mathbb{R}^6$, projects through FC layer to hidden dimension $64$, unfolds across sequence length $T=10$ with a 2-layer GRU, and maps through a final Sigmoid-scaled head:
  $$\hat{q}_t = 30.0 \times \sigma(W_o h_t + b_o)$$
- **Discriminator:** 2-layer Bidirectional GRU processing queue sequences conditioned on $c$, outputting realism probability via BCEWithLogitsLoss.
- **Physical Validation Gate:** Every generated sequence is checked for:
  1. Strict queue boundaries ($0 \le q_{i,t} \le 30$).
  2. Temporal continuity ($|\Delta q_t| \le 12.0$).
  3. Condition-specific queue density alignment.

### Module 5B: Sequence $\beta$-VAE for Incident Management
- **Encoder:** GRU mapping $T=10$ queue history to mean $\mu_z$ and log-variance $\log \sigma^2_z$ ($d_z = 8$).
- **Reparameterization Trick:** $z = \mu_z + \epsilon \odot \exp(0.5 \log \sigma^2_z)$, where $\epsilon \sim \mathcal{N}(0, I_8)$.
- **Decoder:** GRU reconstructing the sequence $\hat{X}_{1:T}$.
- **Loss:** $\mathcal{L}_{\text{VAE}} = \text{MSE}(X, \hat{X}) + \beta D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I))$, with $\beta = 0.05$.
- **Incident Alert Calibration:** Evaluated on normal validation sequences to determine error distribution $(\mu_{\text{err}}, \sigma_{\text{err}})$. Multi-tier thresholding:
  - Warning: $\text{MSE} \ge \mu + 2.0\sigma$
  - Critical: $\text{MSE} \ge \mu + 3.5\sigma$ $\rightarrow$ Triggers alert: **`Critical Traffic Anomaly — Investigation Required`**.

### Signal Controller Verification & Environment Alignment
- **Root Cause of Prior Failure:** In `src/env.py`, `phase_counter` was never reset after initialization, causing the simulation to lock indefinitely in Phase 0 (Action 0 selected 100% of steps). Additionally, lane departure logic was misaligned with the visualizer.
- **Fixes Applied:**
  - Implemented minimum green constraint (`min_green=3` steps per phase) to prevent rapid signal flickering.
  - Reset `phase_counter` whenever an actual phase switch occurs.
  - Aligned physical lanes: Lane 0 = North, Lane 1 = South, Lane 2 = East, Lane 3 = West. Phase 0 discharges lanes {0, 1}; Phase 1 discharges lanes {2, 3}.
  - Rebalanced flow dynamics: Inflow $\sim \mathcal{U}(0, 0.8)$, green discharge rate $0.8$ veh/step, ensuring active lane clearance.
  - Revised reward function: $R = -\text{avg\_queue} - 0.05 \times \text{max\_queue}$, preventing lane starvation.
  - Action distribution post-fix: Balanced across both phases (e.g., 51% Phase 0, 49% Phase 1 in evaluation rollouts).
- **Actuated Signal Logic for Video Uploads:** When processing arbitrary uploaded videos (`upload_mode=True`), the pipeline computes signal phases from observed approach traffic (allocating green to the approach pair with higher queue demand: North-South vs. East-West) while enforcing the strict minimum green duration constraint (`_ACT_MIN_GREEN=3`), count-proportional policy confidence, and zero conflicting greens.

---

## 🖥️ Unified Surveillance Dashboard

The web visualizer (`web_visualizer/visualizer.html`) provides a modern surveillance console styled with light pastel lavender and warm beige surfaces, high-contrast crisp white typography on dark viewports, and fluid CSS animations:
1. **Section 1: Real-Time Traffic Monitoring**  
   Video canvas with bounding boxes, persistent track IDs, dynamic ROI approach counters, non-blocking upload progress bar, and an interactive 32×32 spatial occupancy grid heatmap.
2. **Section 2: Autonomous Signal Management**  
   Four distinct signal heads (North, South, East, West) with active color states, animated pulses, queue height bars, AI phase recommendations, and policy probability gauges.
3. **Section 3: Incident Management & Safety**  
   Real-time VAE reconstruction error gauge, status indicator (**NORMAL**, **WARNING**, **CRITICAL: Critical Traffic Anomaly — Investigation Required**), and a timestamped incident event log.
4. **Section 4: Performance Analytics**  
   Live telemetry tracking average wait time, total queue, throughput, queue reduction percentage, and model benchmark comparisons.

---

## 📁 Repository Structure

```
Autonomous-Traffic-Monitoring-and-Incident-Management-System/
├── README.md                          # Comprehensive system documentation
├── requirements.txt                   # Project dependencies (Torch, OpenCV, etc.)
├── main.py                            # Master CLI entry point
├── server.py                          # Standalone ATMS server launcher
├── src/
│   ├── city_network.py                # 6-intersection city network & emergency dispatch engine
│   ├── env.py                         # 4-way intersection simulation environment
│   ├── model.py                       # MLP policy & Q-network architectures
│   ├── trainer.py                     # Supervised, REINFORCE & DQN training loops
│   ├── evaluate.py                    # Policy evaluation & benchmarking metrics
│   ├── tracker.py                     # Centroid & IOU persistent object tracker
│   ├── cnn_custom.py                  # Custom PyTorch CNN for spatial density
│   ├── vision.py                      # Video ingestion, detection, and grid mapping
│   ├── adapter.py                     # VisionToStateAdapter (7-D standard state)
│   ├── gan.py                         # Conditional Sequence GAN & validation gate
│   ├── vae.py                         # Sequence Beta-VAE & incident anomaly detector
│   ├── temporal.py                    # LSTM/GRU temporal queue forecaster
│   ├── hybrid.py                      # End-to-end hybrid surveillance pipeline
│   └── server.py                      # ThreadingHTTPServer, REST API & video prep
├── models/
│   ├── custom_cnn.pt                  # Trained Custom CNN weights
│   ├── sequence_gan.pt                # Trained Sequence GAN weights
│   ├── sequence_vae.pt                # Trained Sequence VAE weights
│   └── rl_policy.pt                   # Trained RL policy checkpoint
├── results/
│   ├── city_rollout.json              # Multi-intersection city rollout telemetry
│   ├── sample_traffic.mp4             # Synthetic surveillance test video
│   ├── hybrid_rollout.json            # 100-step hybrid rollout telemetry
│   ├── rl_comparison.png              # Benchmark evaluation plot
│   └── supervised_comparison.png      # Supervised performance plot
├── uploads/
│   ├── prepared.mp4                   # Normalized video for fast inference
│   └── debug/                         # Annotated debug frames (moving vs parked)
└── web_visualizer/
    ├── visualizer.html                # Interactive 2D City & CCTV Command Dashboard
    ├── rollout_city.js                # Auto-injected 6-intersection city data
    └── rollout_data.js                # Auto-injected simulation & vision data

```

---

## ⚖️ License & Academic Citation

Developed as an academic demonstration of hybrid artificial intelligence for autonomous traffic monitoring and incident management. All neural architectures adhere to syllabus requirements for Deep Learning.
