import React, { useMemo } from "react";
import { AlphaTimeseries } from "../types";

interface TimeseriesChartsProps {
  data: AlphaTimeseries | null;
  label?: string | null;
}

interface LineSeries {
  label: string;
  data: number[];
  color: string;
}

function buildPath(values: number[], width: number, height: number): string {
  if (!values.length) {
    return "";
  }
  const sanitized: number[] = [];
  let lastFinite = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (const raw of values) {
    const value = Number(raw);
    if (Number.isFinite(value)) {
      lastFinite = value;
      sanitized.push(value);
      if (value < min) min = value;
      if (value > max) max = value;
    } else {
      sanitized.push(lastFinite);
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = -1;
    max = 1;
  }
  if (min === max) {
    const pad = Math.max(1, Math.abs(min) * 0.1 + 0.1);
    min -= pad;
    max += pad;
  }

  const points = sanitized.map((value, index) => {
    const x = (width * index) / Math.max(1, sanitized.length - 1);
    const yRatio = (value - min) / (max - min);
    const y = height - yRatio * height;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return `M${points.join(" L")}`;
}

function LineChart({ series, title }: { series: LineSeries[]; title: string }): React.ReactElement {
  const height = 180;
  const width = 320;
  const paths = useMemo(() => series.map((item) => ({
    color: item.color,
    path: buildPath(item.data, width, height),
    label: item.label,
  })), [series]);

  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <rect x="0" y="0" width={width} height={height} fill="#0f1419" stroke="#1f242a" strokeWidth="1" />
        {paths.map((item) =>
          item.path ? (
            <path
              key={item.label}
              d={item.path}
              fill="none"
              stroke={item.color}
              strokeWidth="2"
            />
          ) : null,
        )}
      </svg>
      <div className="chart-legend">
        {series.map((item) => (
          <span key={item.label} className="chart-legend__item">
            <span className="chart-legend__swatch" style={{ backgroundColor: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function TimeseriesCharts({ data, label }: TimeseriesChartsProps): React.ReactElement {
  if (!data) {
    return (
      <div className="chart-row">
        <div className="chart-card">
          <h3>Equity Curve</h3>
          <p className="muted">Select an alpha to view timeseries.</p>
        </div>
        <div className="chart-card">
          <h3>Returns per Bar</h3>
          <p className="muted">Select an alpha to view timeseries.</p>
        </div>
      </div>
    );
  }

  const equitySeries: LineSeries = {
    label: label ?? "equity",
    data: data.equity,
    color: "#5ab4f0",
  };
  const retSeries: LineSeries = {
    label: label ?? "ret_net",
    data: data.ret_net,
    color: "#f6c560",
  };
  const descriptor = label ? ` — ${label}` : "";

  return (
    <div className="chart-row">
      <LineChart series={[equitySeries]} title={`Equity Curve${descriptor}`} />
      <LineChart series={[retSeries]} title={`Returns per Bar${descriptor}`} />
    </div>
  );
}
