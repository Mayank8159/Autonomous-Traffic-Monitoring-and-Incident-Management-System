import { StatCard } from "@/components/StatCard";
import { DensityGrid } from "@/components/DensityGrid";
import { IncidentCard } from "@/components/IncidentCard";
import { getTracks, getFlow, getDensity, getIncidents, getStatus } from "@/lib/api";
import type { TrackInfo, FlowStats, DensityCell, IncidentInfo, SystemStatus } from "@/types";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  let tracks: TrackInfo[] = [];
  let flow: FlowStats = { entry_count: 0, exit_count: 0, total_vehicles: 0, flow_rate_per_min: 0 };
  let density: DensityCell[] = Array.from({ length: 9 }, (_, i) => ({ row: Math.floor(i / 3), col: i % 3, level: "clear" as const }));
  let incidents: IncidentInfo[] = [];
  let status: SystemStatus = { active_tracks: 0, total_incidents: 0, last_update: Date.now() };

  try {
    [tracks, flow, density, incidents, status] = await Promise.all([
      getTracks(),
      getFlow(),
      getDensity(),
      getIncidents(5),
      getStatus(),
    ]);
  } catch {
    // use defaults
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-white/40 mt-1">Real-time traffic monitoring overview</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-white/30">
          <div className="w-2 h-2 rounded-full bg-traffic-clear animate-pulse" />
          Live
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Active Tracks" value={status.active_tracks} color="blue" pulse />
        <StatCard label="Flow Rate" value={flow.flow_rate_per_min.toFixed(1)} sublabel="/min" color="green" />
        <StatCard label="Total Incidents" value={status.total_incidents} color="red" />
        <StatCard label="Vehicles" value={flow.total_vehicles} sublabel="counted" color="yellow" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <DensityGrid cells={density} />
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
              Vehicle Flow
            </h3>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold text-traffic-clear">{flow.entry_count}</p>
                <p className="text-xs text-white/30 mt-1">Entries</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-traffic-jammed">{flow.exit_count}</p>
                <p className="text-xs text-white/30 mt-1">Exits</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-traffic-info">{flow.flow_rate_per_min.toFixed(0)}</p>
                <p className="text-xs text-white/30 mt-1">Rate/min</p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-3">
              Recent Alerts
            </h3>
            <div className="space-y-2">
              {incidents.length === 0 ? (
                <p className="text-sm text-white/20 glass rounded-lg p-4 text-center">
                  No incidents detected
                </p>
              ) : (
                incidents.map((inc) => (
                  <IncidentCard key={inc.incident_id} incident={inc} />
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
