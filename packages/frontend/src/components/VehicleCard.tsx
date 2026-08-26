import type { TrackInfo } from "@/types";

const classColors: Record<string, string> = {
  car: "border-blue-400/40 bg-blue-400/10",
  truck: "border-orange-400/40 bg-orange-400/10",
  bus: "border-purple-400/40 bg-purple-400/10",
  motorcycle: "border-yellow-400/40 bg-yellow-400/10",
  bicycle: "border-green-400/40 bg-green-400/10",
};

export function VehicleCard({ track }: { track: TrackInfo }) {
  const colorClass = classColors[track.class_name] || "border-gray-400/40 bg-gray-400/10";

  return (
    <div className={`glass rounded-lg p-4 border ${colorClass}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-bold">ID: {track.track_id}</span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 capitalize">
          {track.class_name}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-white/30">Speed</span>
          <p className="font-mono font-bold">{track.speed_kmh.toFixed(1)} km/h</p>
        </div>
        <div>
          <span className="text-white/30">Confidence</span>
          <p className="font-mono font-bold">{(track.confidence * 100).toFixed(0)}%</p>
        </div>
        <div>
          <span className="text-white/30">Position</span>
          <p className="font-mono text-[11px]">
            ({track.center[0]}, {track.center[1]})
          </p>
        </div>
        <div>
          <span className="text-white/30">Status</span>
          <p className={track.is_stationary ? "text-red-400 font-bold" : "text-traffic-clear"}>
            {track.is_stationary ? "STATIONARY" : "MOVING"}
          </p>
        </div>
      </div>
    </div>
  );
}
