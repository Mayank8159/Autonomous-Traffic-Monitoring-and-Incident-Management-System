export interface TrackInfo {
  track_id: number;
  class_name: string;
  bbox: [number, number, number, number];
  center: [number, number];
  speed_kmh: number;
  is_stationary: boolean;
  confidence: number;
  timestamp: number;
}

export interface FlowStats {
  entry_count: number;
  exit_count: number;
  total_vehicles: number;
  flow_rate_per_min: number;
}

export interface DensityCell {
  row: number;
  col: number;
  level: "clear" | "moderate" | "jammed";
}

export interface IncidentInfo {
  incident_id: string;
  timestamp: number;
  type: "stationary_vehicle" | "wrong_way" | "collision" | "overspeed";
  track_id: number;
  location_id: string;
  metadata: Record<string, unknown>;
  snapshot_path?: string;
}

export interface SystemStatus {
  active_tracks: number;
  total_incidents: number;
  last_update: number;
}

export interface SystemConfig {
  model: string;
  confidence_threshold: number;
  speed_limit_kmh: number;
  stationary_threshold_sec: number;
  device: string;
}
