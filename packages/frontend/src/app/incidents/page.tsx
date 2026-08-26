import { IncidentCard } from "@/components/IncidentCard";
import { getIncidents } from "@/lib/api";
import type { IncidentInfo } from "@/types";

export const dynamic = "force-dynamic";

export default async function IncidentsPage() {
  let incidents: IncidentInfo[] = [];
  try {
    incidents = await getIncidents(50);
  } catch {
    // use default
  }

  const byType = incidents.reduce(
    (acc, inc) => {
      acc[inc.type] = (acc[inc.type] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Incidents</h1>
        <p className="text-sm text-white/40 mt-1">Traffic incident detection and alert history</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {(["collision", "wrong_way", "stationary_vehicle", "overspeed"] as const).map((type) => (
          <div key={type} className="glass rounded-xl p-4 text-center">
            <p className="text-2xl font-bold">{byType[type] || 0}</p>
            <p className="text-xs text-white/40 mt-1 capitalize">{type.replace("_", " ")}</p>
          </div>
        ))}
      </div>

      <div className="space-y-3">
        {incidents.length === 0 ? (
          <div className="glass rounded-lg p-12 text-center">
            <svg className="w-12 h-12 mx-auto text-white/10 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-white/20">No incidents recorded</p>
            <p className="text-xs text-white/10 mt-1">System is monitoring normally</p>
          </div>
        ) : (
          incidents.map((inc) => (
            <IncidentCard key={inc.incident_id} incident={inc} />
          ))
        )}
      </div>
    </div>
  );
}
