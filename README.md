<div align="center">

# Autonomous Traffic Monitoring & Incident Management System

### Real-Time Deep Learning Pipeline for Intelligent Traffic Surveillance

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00D4AA?style=for-the-badge&logo=opencv&logoColor=white)](https://ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

```
 INPUT STREAM ──► YOLOv8 DETECTION ──► ByteTrack ──► ANALYTICS ENGINE ──► ALERTS & API
   (RTSP/CCTV)       (CNN Inference)    (Multi-Obj)    (Speed/Collision/     (Webhooks/SMS/
                                      (Tracking)       Density/WrongWay)     Snapshots)
```

</div>

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  VIDEO   │───►│  YOLOv8 CNN │───►│  ByteTrack   │───►│  Analytics    │  │
│  │  SOURCE  │    │  Detector   │    │  MOT System  │    │  Engine       │  │
│  │          │    │  (Backbone) │    │              │    │               │  │
│  │ RTSP/File│    │  n/s/m/l/x  │    │  Kalman +    │    │ Speed Estim.  │  │
│  │ /Webcam  │    │  COCO Weights│    │  Hungarian   │    │ Density Map   │  │
│  └──────────┘    └─────────────┘    └──────────────┘    │ Wrong-Way Det.│  │
│       │              │                    │               │ Collision Det.│  │
│       │              │                    │               │ Flow Counter  │  │
│       │              │                    │               │ Stationary Det│  │
│       │              │                    │               └───────┬───────┘  │
│       │              │                    │                       │          │
│       │              │                    │               ┌───────▼───────┐  │
│       │              │                    │               │ Alert System  │  │
│       │              │                    │               │ • Snapshots   │  │
│       │              │                    │               │ • JSON Logger │  │
│       │              │                    │               │ • Webhooks    │  │
│       │              │                    │               │ • SMS (Mock)  │  │
│       │              │                    │               └───────┬───────┘  │
│       │              │                    │                       │          │
│  ┌────▼──────────────▼────────────────────▼───────────────────────▼───────┐  │
│  │                    Visual Overlay Renderer (OpenCV)                     │  │
│  │  • Bounding Boxes  • Trajectories  • Speed Vectors  • Density Heatmap  │  │
│  │  • Counting Line   • HUD Overlay   • Incident Banners                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                            ┌───────▼───────┐                                 │
│                            │  FastAPI JSON  │                                │
│                            │  Telemetry     │                                │
│                            │  Server        │                                │
│                            │  (Background)  │                                │
│                            └───────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Features

### Detection & Tracking Pipeline

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Object Detection | **YOLOv8** (COCO pretrained) | Vehicle localization & classification |
| Multi-Object Tracking | **ByteTrack** | Persistent track IDs across frames |
| Supported Classes | Car, Truck, Bus, Motorcycle, Bicycle | COCO vehicle classes only |
| Inference Modes | FP16 / GPU / CPU | Configurable device & precision |

### Analytics Engine

| Module | Method | Output |
|--------|--------|--------|
| **Speed Estimation** | Perspective transform homography + temporal smoothing | Per-vehicle km/h |
| **Density Mapping** | Grid-based occupancy ratio (3x3 configurable) | CLEAR / MODERATE / JAMMED |
| **Vehicle Counting** | Directional counting line intersection | Entry/Exit counts + flow rate |
| **Stationary Detection** | Speed epsilon threshold over time window | Breakdown hazard alerts |
| **Wrong-Way Detection** | Movement vector vs. lane vector angle comparison | Directional violation alerts |
| **Collision Detection** | IoU spike analysis + deceleration + bbox rotation | Multi-signal collision alerts |

### Alert & Notification System

- **Snapshot Export** -- Annotated JPEG frames saved to disk with incident metadata in filename
- **Structured Logging** -- Daily JSONL incident logs with UTC timestamps
- **Webhook Dispatch** -- Async HTTP POST to configurable endpoints with semaphore-controlled concurrency
- **SMS Alerts** -- Mock SMS gateway for high-severity incidents (collision, wrong-way)
- **Recent Alert Buffer** -- In-memory ring buffer of last 200 alerts for API access

### Live Telemetry API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /health` | Service health status |
| `GET /api/tracks` | All currently tracked vehicles with speed, bbox, class |
| `GET /api/flow` | Vehicle flow statistics (entry, exit, rate/min) |
| `GET /api/density` | Grid-based congestion map |
| `GET /api/incidents` | Recent incident/alert history |
| `GET /api/status` | System status (active tracks, total incidents, FPS) |
| `GET /api/config` | Non-sensitive configuration dump |

---

## Project Structure

```
Autonomous-Traffic-Monitoring-and-Incident-Management-System/
│
├── main.py              # Entry-point: stream loop, CLI, orchestration
├── detector.py          # YOLOv8 + ByteTrack detection & tracking pipeline
├── analytics.py         # Speed, density, counting, collision, wrong-way analytics
├── alert_system.py      # Snapshot export, logging, webhooks, SMS dispatch
├── api.py               # FastAPI telemetry server (background thread)
├── config.py            # Central dataclass-based configuration system
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended for real-time inference)
- RTSP camera stream, video file, or webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/Mayank8159/Autonomous-Traffic-Monitoring-and-Incident-Management-System.git
cd Autonomous-Traffic-Monitoring-and-Incident-Management-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run with default webcam (source=0)
python main.py

# Run with RTSP stream
python main.py --source "rtsp://admin:password@192.168.1.100:554/stream1"

# Run with video file
python main.py --source "traffic_sample.mp4"

# Run headless (no display window)
python main.py --source "rtsp://..." --no-display
```

### CLI Options

```
python main.py [OPTIONS]

Options:
  -s, --source TEXT       Video source: RTSP URL, file path, or camera index
                          [default: 0]
  -m, --model TEXT        YOLOv8 model name/path  [default: yolov8n.pt]
  --conf FLOAT            Detection confidence threshold  [default: 0.45]
  --device TEXT           Inference device (cpu / 0 / 0,1)  [default: 0]
  --speed-limit FLOAT     Speed limit in km/h  [default: 60.0]
  --no-display            Run headless without OpenCV window
```

### Access the API

Once running, the telemetry API is available at:

```
http://localhost:8000
http://localhost:8000/docs    # Interactive Swagger UI
```

---

## Configuration

All system parameters are defined in `config.py` via typed dataclasses. Key configuration groups:

| Config Group | Key Parameters |
|-------------|----------------|
| **StreamConfig** | source, frame_width (1280), frame_height (720), buffer_size |
| **DetectionConfig** | model_name, confidence_threshold (0.45), iou_threshold (0.5), half_precision |
| **TrackerConfig** | track_thresh (0.5), track_buffer (60), match_thresh (0.8), min_hits (3) |
| **SpeedConfig** | calibration_points, pixels_to_meters_ratio, speed_limit_kmh (60), smoothing_window |
| **StationaryConfig** | stationary_threshold_sec (5.0), stationary_speed_epsilon (2.0) |
| **WrongWayConfig** | lane_vectors, angle_tolerance_deg (90), min_track_length (10) |
| **CollisionConfig** | iou_spike_threshold (0.3), deceleration_threshold_kmh (30), bbox_orientation_shift_deg (30) |
| **VisualizationConfig** | show_bounding_boxes, show_trajectories, show_speed_vectors, show_density_heatmap |

---

## Visualization

The renderer produces a real-time annotated overlay including:

- **Bounding Boxes** -- Color-coded by vehicle class (car=blue, truck=orange, bus=magenta, motorcycle=yellow, bicycle=green)
- **Trajectory Trails** -- Fading line segments showing recent movement history
- **Speed Vectors** -- Arrowed lines indicating instantaneous velocity direction
- **Density Heatmap** -- Semi-transparent grid overlay (green=clear, orange=moderate, red=jammed)
- **Counting Line** -- Yellow line with label for vehicle flow measurement
- **HUD Panel** -- FPS, frame count, active tracks, entry/exit counts, flow rate
- **Incident Banners** -- Red alert banners for detected incidents

---

## Neural Network Pipeline Details

```
                          YOLOv8 INFERENCE PIPELINE
                    ┌─────────────────────────────────┐
                    │                                 │
  Input Frame       │   ┌─────────┐                  │
  (1280x720)  ─────►│   │ Backbone │  CSPDarknet      │
                    │   │  (CSP)  │  Feature Extractor│
                    │   └────┬────┘                  │
                    │        │                        │
                    │   ┌────▼────┐                  │
                    │   │  Neck   │  PANet + FPN      │
                    │   │ (PANet) │  Multi-Scale       │
                    │   └────┬────┘  Feature Fusion   │
                    │        │                        │
                    │   ┌────▼────┐                  │
                    │   │  Head   │  Anchor-Free       │
                    │   │ (Detect)│  Detection Head    │
                    │   └────┬────┘                  │
                    │        │                        │
                    │   ┌────▼────┐                  │
                    │   │   NMS   │  Non-Maximum       │
                    │   │ + Filter│  Suppression       │
                    │   └────┬────┘  (IoU=0.5)        │
                    │        │                        │
                    └────────│─────────────────────────┘
                             │
                    Detections: xyxy + conf + class
                             │
                    ┌────────▼──────────────────────┐
                    │        ByteTrack               │
                    │   ┌──────────┐                │
                    │   │ Kalman   │  State           │
                    │   │ Filter   │  Prediction      │
                    │   └────┬─────┘                │
                    │        │                       │
                    │   ┌────▼─────┐                │
                    │   │Hungarian │  Assignment      │
                    │   │ Matching │  Algorithm       │
                    │   └────┬─────┘                │
                    │        │                       │
                    │   ┌────▼─────┐                │
                    │   │ Track    │  ID              │
                    │   │ Management│ Persistence     │
                    │   └──────────┘                │
                    └───────────────────────────────┘
                             │
                    Tracked: (track_id, class, bbox, conf)
                             │
                    ┌────────▼──────────────────────┐
                    │      ANALYTICS ENGINE          │
                    │                                │
                    │  Speed ──► Perspective ──► km/h│
                    │  Density ──► Grid ──► Congest. │
                    │  Counter ──► Line Cross ──► Flow│
                    │  Stationary ──► Duration ──► Hz│
                    │  WrongWay ──► Vector Angle ──► │
                    │  Collision ──► IoU Spike ──►   │
                    └───────────────────────────────┘
```

---

## Technology Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Language | Python | 3.9+ |
| Deep Learning | PyTorch | 2.0+ |
| Object Detection | Ultralytics YOLOv8 | 8.0+ |
| Multi-Object Tracking | ByteTrack (boxmot) | 10.0+ |
| Computer Vision | OpenCV | 4.8+ |
| Array Computing | NumPy | 1.24+ |
| Web Framework | FastAPI | 0.100+ |
| ASGI Server | Uvicorn | 0.23+ |
| Async HTTP | aiohttp | 3.9+ |
| Data Validation | Pydantic | 2.0+ |
| Geometry | Shapely | 2.0+ |

---

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for intelligent transportation systems research and deployment.**

</div>
