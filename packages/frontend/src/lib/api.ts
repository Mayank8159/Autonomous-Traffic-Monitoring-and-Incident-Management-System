import type { TrackInfo, FlowStats, DensityCell, IncidentInfo, SystemStatus, SystemConfig } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";

async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { next: { revalidate: 0 } });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getTracks(): Promise<TrackInfo[]> {
  return fetchAPI<TrackInfo[]>("/api/tracks");
}

export async function getFlow(): Promise<FlowStats> {
  return fetchAPI<FlowStats>("/api/flow");
}

export async function getDensity(): Promise<DensityCell[]> {
  return fetchAPI<DensityCell[]>("/api/density");
}

export async function getIncidents(limit = 20): Promise<IncidentInfo[]> {
  return fetchAPI<IncidentInfo[]>(`/api/incidents?limit=${limit}`);
}

export async function getStatus(): Promise<SystemStatus> {
  return fetchAPI<SystemStatus>("/api/status");
}

export async function getConfig(): Promise<SystemConfig> {
  return fetchAPI<SystemConfig>("/api/config");
}
