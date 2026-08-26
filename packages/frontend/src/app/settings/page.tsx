import { getConfig } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function SettingsPage() {
  let config;
  try {
    config = await getConfig();
  } catch {
    config = null;
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-white/40 mt-1">System configuration and detection parameters</p>
      </div>

      <div className="glass rounded-xl p-6 space-y-6">
        <div>
          <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
            Detection Model
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-white/30 block mb-1">Model</label>
              <div className="glass rounded-lg px-3 py-2 text-sm font-mono">
                {config?.model || "yolov8n.pt"}
              </div>
            </div>
            <div>
              <label className="text-xs text-white/30 block mb-1">Device</label>
              <div className="glass rounded-lg px-3 py-2 text-sm font-mono">
                {config?.device || "cpu"}
              </div>
            </div>
            <div>
              <label className="text-xs text-white/30 block mb-1">Confidence Threshold</label>
              <div className="glass rounded-lg px-3 py-2 text-sm font-mono">
                {config?.confidence_threshold || 0.45}
              </div>
            </div>
            <div>
              <label className="text-xs text-white/30 block mb-1">Speed Limit</label>
              <div className="glass rounded-lg px-3 py-2 text-sm font-mono">
                {config?.speed_limit_kmh || 60} km/h
              </div>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
            Incident Detection
          </h3>
          <div className="space-y-3">
            {[
              { label: "Stationary Vehicle Threshold", value: `${config?.stationary_threshold_sec || 5}s` },
              { label: "Wrong-Way Angle Tolerance", value: "90deg" },
              { label: "Collision IoU Spike Threshold", value: "0.3" },
              { label: "Overspeed Detection", value: `>${config?.speed_limit_kmh || 60} km/h` },
            ].map((item) => (
              <div key={item.label} className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-sm text-white/60">{item.label}</span>
                <span className="text-sm font-mono text-white/40">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
            Backend Architecture
          </h3>
          <div className="space-y-2 text-sm text-white/40">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-traffic-clear" />
              AWS Lambda (Python 3.11)
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-traffic-clear" />
              API Gateway + DynamoDB + S3
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-traffic-clear" />
              SAM (Serverless Application Model)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
