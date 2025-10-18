import React from "react";
import { SPARKLINE_HEIGHT, SPARKLINE_WIDTH } from "./utils";

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
    <div className="pipeline-activity__metrics">
      {metrics.map((metric) => (
        <div key={metric.id} className="pipeline-activity__metric">
          <span className="pipeline-activity__metric-label">{metric.label}</span>
          <span className="pipeline-activity__metric-value">{metric.value}</span>
          {metric.delta ? (
            <span
              className={`metric-delta ${
                metric.trend === "up"
                  ? "metric-delta--up"
                  : metric.trend === "down"
                    ? "metric-delta--down"
                    : ""
              }`}
            >
              {metric.delta}
            </span>
          ) : null}
          {metric.caption ? <span className="pipeline-activity__metric-caption muted">{metric.caption}</span> : null}
        </div>
      ))}

      <div className="pipeline-activity__metric pipeline-activity__metric--sparkline">
        <span className="pipeline-activity__metric-label">Fitness trend</span>
        {sparkline ? (
          <>
            <svg
              className="pipeline-activity__sparkline"
              viewBox={`0 0 ${SPARKLINE_WIDTH} ${SPARKLINE_HEIGHT}`}
              xmlns="http://www.w3.org/2000/svg"
              preserveAspectRatio="none"
            >
              <polyline points={sparkline.points} />
            </svg>
            <div className="pipeline-activity__sparkline-meta muted">
              <span>Peak {sparkline.max.toFixed(3)}</span>
              <span>Floor {sparkline.min.toFixed(3)}</span>
            </div>
          </>
        ) : (
          <span className="muted">Collecting data…</span>
        )}
      </div>
    </div>
  );
}
