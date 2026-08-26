import { VehicleCard } from "@/components/VehicleCard";
import { getTracks } from "@/lib/api";
import type { TrackInfo } from "@/types";

export const dynamic = "force-dynamic";

export default async function TrackingPage() {
  let tracks: TrackInfo[] = [];
  try {
    tracks = await getTracks();
  } catch {
    // use default
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Live Tracking</h1>
        <p className="text-sm text-white/40 mt-1">Real-time vehicle detection and tracking</p>
      </div>

      <div className="glass rounded-xl p-6">
        <div className="aspect-video bg-black/40 rounded-lg border border-white/5 flex items-center justify-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-traffic-info/5 to-transparent" />
          <div className="text-center relative z-10">
            <svg className="w-16 h-16 mx-auto text-white/10 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <p className="text-sm text-white/20">Live video feed</p>
            <p className="text-xs text-white/10 mt-1">
              Connect an RTSP source or upload frames to S3
            </p>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-3">
          Detected Vehicles ({tracks.length})
        </h3>
        {tracks.length === 0 ? (
          <div className="glass rounded-lg p-8 text-center">
            <p className="text-sm text-white/20">No vehicles currently tracked</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {tracks.map((track) => (
              <VehicleCard key={track.track_id} track={track} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
