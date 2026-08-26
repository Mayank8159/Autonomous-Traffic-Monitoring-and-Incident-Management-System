import { clsx } from "clsx";

interface StatCardProps {
  label: string;
  value: string | number;
  sublabel?: string;
  color?: "blue" | "green" | "red" | "yellow" | "gray";
  pulse?: boolean;
}

const colorMap = {
  blue: "text-traffic-info",
  green: "text-traffic-clear",
  red: "text-traffic-jammed",
  yellow: "text-traffic-moderate",
  gray: "text-white/40",
};

export function StatCard({ label, value, sublabel, color = "blue", pulse }: StatCardProps) {
  return (
    <div className="glass rounded-xl p-5">
      <p className="text-xs font-medium text-white/40 uppercase tracking-wider">{label}</p>
      <div className="flex items-end gap-2 mt-2">
        <span className={clsx("text-3xl font-bold tabular-nums", colorMap[color], pulse && "animate-pulse-slow")}>
          {value}
        </span>
        {sublabel && <span className="text-xs text-white/30 mb-1">{sublabel}</span>}
      </div>
    </div>
  );
}
