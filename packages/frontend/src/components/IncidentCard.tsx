import { clsx } from "clsx";
import type { IncidentInfo } from "@/types";

const typeColors: Record<string, string> = {
  collision: "bg-red-500/20 text-red-400 border-red-500/30",
  wrong_way: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  stationary_vehicle: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  overspeed: "bg-blue-500/20 text-blue-400 border-blue-500/30",
};

const typeIcons: Record<string, string> = {
  collision: "!",
  wrong_way: "~",
  stationary_vehicle: "||",
  overspeed: ">>",
};

export function IncidentCard({ incident }: { incident: IncidentInfo }) {
  const ts = new Date(incident.timestamp * 1000);
  const colorClass = typeColors[incident.type] || "bg-gray-500/20 text-gray-400 border-gray-500/30";

  return (
    <div className="glass rounded-lg p-4 hover:bg-white/[0.02] transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={clsx("w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold border", colorClass)}>
            {typeIcons[incident.type] || "?"}
          </div>
          <div>
            <p className="text-sm font-semibold capitalize">{incident.type.replace("_", " ")}</p>
            <p className="text-xs text-white/40 mt-0.5">
              Track #{incident.track_id} &middot; {incident.location_id}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs text-white/30">
            {ts.toLocaleTimeString()}
          </p>
          <p className="text-xs text-white/20">
            {ts.toLocaleDateString()}
          </p>
        </div>
      </div>
      {incident.metadata && Object.keys(incident.metadata).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(incident.metadata).slice(0, 3).map(([k, v]) => (
            <span key={k} className="text-[10px] px-2 py-0.5 rounded bg-white/5 text-white/40">
              {k}: {String(v)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
