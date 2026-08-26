<div align="center">

# Autonomous Traffic Monitoring & Incident Management System

### Serverless Monorepo — AWS Lambda + Next.js + YOLOv8

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![AWS Lambda](https://img.shields.io/badge/AWS_Lambda-Serverless-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/lambda/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-00D4AA?style=for-the-badge)](https://ultralytics.com/)
[![Turborepo](https://img.shields.io/badge/Turborepo-Monorepo-EA35F0?style=for-the-badge)](https://turbo.build/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  S3 Upload  │────►│ Lambda:Detect│────►│ Lambda:     │────►│ DynamoDB     │
│  (Frames)   │     │ (YOLOv8)     │     │ Analytics   │     │ (Tracks +    │
└─────────────┘     └──────────────┘     └─────────────┘     │  Incidents)  │
                                                              └──────┬───────┘
┌─────────────┐     ┌──────────────┐     ┌─────────────┐            │
│  Vercel     │────►│ Next.js 14   │────►│ API Gateway │────────────┘
│  (Frontend) │     │ Dashboard    │     │ (REST API)  │
└─────────────┘     └──────────────┘     └─────────────┘
```

</div>

---

## Architecture

```
                        MONOREPO STRUCTURE
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   packages/backend/                        │  │
│  │                                                            │  │
│  │  template.yaml    ─── AWS SAM IaC (Lambda + API GW +      │  │
│  │                         DynamoDB + S3)                     │  │
│  │  app/                                                   │  │
│  │    ├── config.py         ─── System configuration         │  │
│  │    ├── models.py         ─── Pydantic data models         │  │
│  │    ├── detector.py       ─── YOLOv8 + ByteTrack           │  │
│  │    ├── analytics.py      ─── Speed / Density / Collision  │  │
│  │    ├── handlers/                                          │  │
│  │    │   ├── detect.py     ─── S3 trigger → detection       │  │
│  │    │   ├── analytics.py  ─── Scheduled analytics run      │  │
│  │    │   ├── alerts.py     ─── DynamoDB Stream → webhooks   │  │
│  │    │   └── tracks.py     ─── REST API endpoints           │  │
│  │    └── utils/            ─── DynamoDB / S3 / response     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  packages/frontend/                        │  │
│  │                                                            │  │
│  │  src/                                                     │  │
│  │    ├── app/                                                │  │
│  │    │   ├── layout.tsx    ─── Root layout + sidebar         │  │
│  │    │   ├── page.tsx      ─── Dashboard overview            │  │
│  │    │   ├── tracking/     ─── Live vehicle tracking         │  │
│  │    │   ├── incidents/    ─── Incident history              │  │
│  │    │   ├── analytics/    ─── Speed / density charts        │  │
│  │    │   └── settings/     ─── Config viewer                 │  │
│  │    ├── components/      ─── Sidebar, StatCard, etc.        │  │
│  │    ├── lib/api.ts       ─── API client                     │  │
│  │    └── types/index.ts   ─── TypeScript interfaces          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 packages/shared/                           │  │
│  │  index.ts              ─── Shared TypeScript types          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Serverless Data Flow

```
     Edge Device / RTSP Stream
            │
            ▼
  ┌──────────────────┐
  │  Frame Capture   │  Extract frames → upload to S3
  │  (Python/OpenCV) │  s3://snapshots-dev/frames/*.jpg
  └────────┬─────────┘
           │ S3:ObjectCreated:* trigger
           ▼
  ┌──────────────────┐
  │  Lambda:Detect   │  YOLOv8 inference + ByteTrack
  │  (3008 MB)       │  → Write track data to DynamoDB
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Lambda:Analytics │  Speed estimation, density mapping
  │ (Scheduled 1min) │  collision detection, wrong-way
  │                  │  → Write incidents to IncidentTable
  └────────┬─────────┘
           │ DynamoDB Stream trigger
           ▼
  ┌──────────────────┐
  │  Lambda:Alerts   │  Webhook dispatch, SMS mock
  │  (512 MB)        │  Structured logging
  └──────────────────┘
           │
           ▼
  ┌──────────────────┐     ┌──────────────────┐
  │  API Gateway     │◄────│  Lambda:Tracks   │  REST API for
  │  (REST API)      │     │  (512 MB)        │  frontend queries
  └────────┬─────────┘     └──────────────────┘
           │
           ▼
  ┌──────────────────┐
  │  Next.js 14      │  Dashboard, tracking, incidents
  │  (Vercel)        │  analytics, density maps
  └──────────────────┘
```

---

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- AWS CLI configured (`aws configure`)
- AWS SAM CLI (`pip install aws-sam-cli`)
- Vercel CLI (`npm i -g vercel`)

### Installation

```bash
git clone https://github.com/Mayank8159/Autonomous-Traffic-Monitoring-and-Incident-Management-System.git
cd Autonomous-Traffic-Monitoring-and-Incident-Management-System

# Install monorepo dependencies
npm install

# Install backend Python dependencies
cd packages/backend
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
cd ../..
```

### Deploy Backend (AWS)

```bash
cd packages/backend

# Build and deploy with SAM
sam build
sam deploy --guided

# Note the API Gateway URL from stack outputs
cd ../..
```

### Deploy Frontend (Vercel)

```bash
cd packages/frontend

# Set environment variable
echo "NEXT_PUBLIC_API_URL=<YOUR_API_GATEWAY_URL>" > .env.local

# Deploy to Vercel
vercel deploy --prod

cd ../..
```

### Local Development

```bash
# Start frontend dev server (port 3000)
npm run dev

# Start backend local API (port 3001)
cd packages/backend
sam local start-api --port 3001
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/tracks` | All currently tracked vehicles |
| `GET` | `/api/flow` | Vehicle flow statistics |
| `GET` | `/api/density` | Grid-based congestion map |
| `GET` | `/api/incidents` | Recent incident history |
| `GET` | `/api/status` | System status |
| `GET` | `/api/config` | Configuration dump |

---

## Frontend Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | Real-time overview with stats, density map, alerts |
| `/tracking` | Live Tracking | Vehicle cards with speed, position, status |
| `/incidents` | Incidents | Incident history with type breakdowns |
| `/analytics` | Analytics | Speed distribution, vehicle classes, density grid |
| `/settings` | Settings | System configuration viewer |

---

## AWS Resources

| Resource | Type | Purpose |
|----------|------|---------|
| `DetectFunction` | Lambda (3008 MB) | YOLOv8 inference on uploaded frames |
| `AnalyticsFunction` | Lambda (1024 MB) | Speed, density, collision analytics |
| `AlertFunction` | Lambda (512 MB) | Webhook/SMS alert dispatch |
| `TracksFunction` | Lambda (512 MB) | REST API for telemetry data |
| `TrafficTable` | DynamoDB | Vehicle track data with GSIs |
| `IncidentTable` | DynamoDB | Incident records with GSIs |
| `SnapshotBucket` | S3 | Frame snapshots (30-day lifecycle) |
| `TrafficApi` | API Gateway | REST API with CORS |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Monorepo | Turborepo + npm workspaces |
| Backend | Python 3.11, AWS Lambda, SAM |
| Detection | YOLOv8 (Ultralytics), ByteTrack (boxmot) |
| Database | Amazon DynamoDB (on-demand) |
| Storage | Amazon S3 |
| API | API Gateway REST + Lambda |
| Frontend | Next.js 14, React 18, TypeScript |
| Styling | Tailwind CSS (dark theme) |
| Charts | Recharts |
| Deploy | AWS SAM (backend), Vercel (frontend) |

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with serverless architecture for scalable, cost-efficient traffic surveillance.**

</div>
