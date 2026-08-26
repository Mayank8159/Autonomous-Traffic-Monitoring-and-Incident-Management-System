import { getFlow, getDensity, getTracks, getStatus } from "@/lib/api";
import type { FlowStats, DensityCell, TrackInfo, SystemStatus } from "@/types";

export const dynamic = "force-dynamic";

export default async function AnalyticsPage() {
  let flow: FlowStats = { entry_count: 0, exit_count: 0, total_vehicles: 0, flow_rate_per_min: 0 };
  let density: DensityCell[] = Array.from({ length: 9 }, (_, i) => ({ row: Math.floor(i / 3), col: i % 3, level: "clear" as const }));
  let tracks: TrackInfo[] = [];
  let status: SystemStatus = { active_tracks: 0, total_incidents: 0, last_update: Date.now() };
  try {
    [flow, density, tracks, status] = await Promise.all([
      getFlow(),
      getDensity(),
      getTracks(),
      getStatus(),
    ]);
  } catch {
    // use defaults
  }

  const speedBuckets = [0, 0, 0, 0];
  tracks.forEach((t) => {
    if (t.speed_kmh < 20) speedBuckets[0]++;
    else if (t.speed_kmh < 40) speedBuckets[1]++;
    else if (t.speed_kmh < 60) speedBuckets[2]++;
    else speedBuckets[3]++;
  });

  const classCounts: Record<string, number> = {};
  tracks.forEach((t) => {
    classCounts[t.class_name] = (classCounts[t.class_name] || 0) + 1;
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Analytics</h1>
        <p className="text-sm text-white/40 mt-1">Traffic flow analysis and vehicle statistics</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
            Speed Distribution
          </h3>
          <div className="flex items-end gap-3 h-40">
            {["0-20", "20-40", "40-60", "60+"].map((label, i) => {
              const max = Math.max(...speedBuckets, 1);
              const pct = (speedBuckets[i] / max) * 100;
              const colors = ["bg-traffic-clear", "bg-traffic-info", "bg-traffic-moderate", "bg-traffic-jammed"];
              return (
                <div key={label} className="flex-1 flex flex-col items-center gap-2">
                  <span className="text-xs font-mono text-white/30">{speedBuckets[i]}</span>
                  <div className="w-full relative" style={{ height: `${Math.max(pct, 5)}%` }}>
                    <div className={`absolute inset-0 rounded-t ${colors[i]} opacity-60`} />
                  </div>
                  <span className="text-[10px] text-white/30">{label}</span>
                </div>
              );
            })}
          </div>
          <p className="text-[10px] text-white/20 text-center mt-3">Speed (km/h)</p>
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
            Vehicle Classes
          </h3>
          <div className="space-y-3">
            {Object.entries(classCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, count]) => {
                const max = Math.max(...Object.values(classCounts), 1);
                const pct = (count / max) * 100;
                return (
                  <div key={cls}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="capitalize text-white/60">{cls}</span>
                      <span className="font-mono text-white/30">{count}</span>
                    </div>
                    <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-traffic-info/60 rounded-full" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            {Object.keys(classCounts).length === 0 && (
              <p className="text-sm text-white/20 text-center py-4">No vehicle data</p>
            )}
          </div>
        </div>
      </div>

      <div className="glass rounded-xl p-5">
        <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
          Density Grid Analysis
        </h3>
        <div className="grid grid-cols-3 gap-2 max-w-md">
          {density.map((cell) => {
            const colors = {
              clear: "bg-traffic-clear/20 text-traffic-clear",
              moderate: "bg-traffic-moderate/20 text-traffic-moderate",
              jammed: "bg-traffic-jammed/20 text-traffic-jammed",
            };
            return (
              <div
                key={`${cell.row}-${cell.col}`}
                className={`aspect-square rounded-lg flex items-center justify-center text-xs font-bold ${colors[cell.level]}`}
              >
                {cell.level.toUpperCase()}
              </div>
            );
          })}
        </div>
      </div>

      <div className="glass rounded-xl p-5">
        <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-3">
          System Metrics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-white/30">Active Tracks</p>
            <p className="text-lg font-bold font-mono">{status.active_tracks}</p>
          </div>
          <div>
            <p className="text-xs text-white/30">Flow Rate</p>
            <p className="text-lg font-bold font-mono">{flow.flow_rate_per_min.toFixed(1)}/min</p>
          </div>
          <div>
            <p className="text-xs text-white/30">Total Incidents</p>
            <p className="text-lg font-bold font-mono">{status.total_incidents}</p>
          </div>
          <div>
            <p className="text-xs text-white/30">Total Vehicles</p>
            <p className="text-lg font-bold font-mono">{flow.total_vehicles}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
