"use client";

import { clsx } from "clsx";
import type { DensityCell } from "@/types";

const levelColors: Record<string, string> = {
  clear: "bg-traffic-clear/30 border-traffic-clear/20",
  moderate: "bg-traffic-moderate/30 border-traffic-moderate/20",
  jammed: "bg-traffic-jammed/30 border-traffic-jammed/20",
};

const levelLabels: Record<string, string> = {
  clear: "CLEAR",
  moderate: "MOD",
  jammed: "JAM",
};

export function DensityGrid({ cells }: { cells: DensityCell[] }) {
  const grid: DensityCell[][] = [[], [], []];
  cells.forEach((c) => {
    if (c.row < 3 && c.col < 3) grid[c.row][c.col] = c;
  });

  return (
    <div className="glass rounded-xl p-5">
      <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-4">
        Traffic Density Map
      </h3>
      <div className="grid grid-cols-3 gap-2">
        {grid.map((row, r) =>
          row.map((cell, c) => (
            <div
              key={`${r}-${c}`}
              className={clsx(
                "aspect-video rounded-lg border flex items-center justify-center text-xs font-bold",
                levelColors[cell?.level || "clear"]
              )}
            >
              {levelLabels[cell?.level || "clear"]}
            </div>
          ))
        )}
      </div>
      <div className="flex justify-center gap-4 mt-4 text-[10px] text-white/30">
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded bg-traffic-clear" /> Clear
        </span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded bg-traffic-moderate" /> Moderate
        </span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded bg-traffic-jammed" /> Jammed
        </span>
      </div>
    </div>
  );
}
