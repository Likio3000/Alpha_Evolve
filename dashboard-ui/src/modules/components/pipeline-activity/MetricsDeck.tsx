import React from "react";
import { SPARKLINE_HEIGHT, SPARKLINE_WIDTH } from "./utils";
import { cn } from "@/lib/utils";

export interface MetricEntry {
  id: string;
  label: string;
  value: string;
  delta?: string | null;
  trend?: "up" | "down" | null;
  caption?: string | null;
}

interface MetricsDeckProps {
  metrics: MetricEntry[];
  sparkline?: { points: string; min: number; max: number } | null;
}

export function MetricsDeck({ metrics, sparkline }: MetricsDeckProps): React.ReactElement {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric) => (
        <div key={metric.id} className="flex flex-col p-3 rounded bg-secondary/20 border border-secondary/50">
          <span className="text-xs font-medium text-muted-foreground uppercase">{metric.label}</span>
          <span className="text-2xl font-bold tracking-tight my-1">{metric.value}</span>
          {metric.delta ? (
            <span
              className={cn(
                "text-xs font-semibold px-1.5 py-0.5 rounded w-fit",
                metric.trend === "up" ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400" :
                  metric.trend === "down" ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400" :
                    "bg-muted text-muted-foreground"
              )}
            >
              {metric.delta}
            </span>
          ) : null}
          {metric.caption ? <span className="text-[10px] text-muted-foreground mt-1">{metric.caption}</span> : null}
        </div>
      ))}

      <div className="col-span-2 lg:col-span-1 flex flex-col p-3 rounded bg-secondary/20 border border-secondary/50">
        <span className="text-xs font-medium text-muted-foreground uppercase mb-2">Fitness trend</span>
        {sparkline ? (
          <div className="flex-1 flex flex-col justify-end">
            <svg
              className="w-full h-[32px] overflow-visible"
              viewBox={`0 0 ${SPARKLINE_WIDTH} ${SPARKLINE_HEIGHT}`}
              xmlns="http://www.w3.org/2000/svg"
              preserveAspectRatio="none"
              stroke="currentColor"
              fill="none"
              strokeWidth="2"
            >
              <polyline points={sparkline.points} className="text-primary" />
            </svg>
            <div className="flex justify-between text-[10px] text-muted-foreground mt-2">
              <span>Peak {sparkline.max.toFixed(3)}</span>
              <span>Floor {sparkline.min.toFixed(3)}</span>
            </div>
          </div>
        ) : (
          <span className="text-sm text-muted-foreground italic h-full flex items-center">Collecting data…</span>
        )}
      </div>
    </div>
  );
}
