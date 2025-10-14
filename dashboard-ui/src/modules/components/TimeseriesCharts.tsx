import React, { useMemo } from "react";
import type { AlphaTimeseries } from "../types";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TooltipProps } from "recharts";

interface TimeseriesChartsProps {
  data: AlphaTimeseries | null;
  label?: string | null;
}

interface ChartRow {
  index: number;
  date: string;
  axisLabel: string;
  tooltipLabel: string;
  equity: number | null;
  ret_net: number | null;
}

interface SanitizedTimeseries {
  rows: ChartRow[];
  hasEquity: boolean;
  hasReturns: boolean;
  equityDomain: [number, number];
  returnsDomain: [number, number];
}

const COLORS = {
  equityStroke: "#5ab4f0",
  equityFillStart: "rgba(90, 180, 240, 0.55)",
  equityFillEnd: "rgba(90, 180, 240, 0.05)",
  grid: "rgba(90, 180, 240, 0.12)",
  axis: "rgba(167, 173, 179, 0.7)",
  tooltipBorder: "rgba(90, 180, 240, 0.45)",
  tooltipBg: "rgba(10, 14, 20, 0.92)",
  retPositive: "#6ee7b7",
  retNegative: "#ef6d7a",
  retNeutral: "#f6c560",
};

const YAXIS_PROPS = {
  stroke: COLORS.axis,
  tick: { fill: COLORS.axis, fontSize: 11 },
  axisLine: false,
  tickLine: false,
};

const XAXIS_PROPS = {
  stroke: COLORS.axis,
  tick: { fill: COLORS.axis, fontSize: 11 },
  axisLine: false,
  tickLine: false,
  minTickGap: 12,
};

function sanitizeNumber(value: unknown): number | null {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function padDomain(value: number): number {
  return Math.max(1, Math.abs(value) * 0.1);
}

function buildAxisLabel(raw: string, index: number): string {
  if (!raw) {
    return `#${index + 1}`;
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return raw;
  }
  return parsed.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function buildTooltipLabel(raw: string, index: number): string {
  if (!raw) {
    return `Bar ${index + 1}`;
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return raw;
  }
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatEquity(value: number | null): string {
  if (value == null) return "—";
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: Math.abs(value) >= 100 ? 0 : 2,
  }).format(value);
}

function formatReturn(value: number | null): string {
  if (value == null) return "—";
  const percent = value * 100;
  const digits = Math.abs(percent) < 0.1 ? 3 : 2;
  return `${percent.toFixed(digits)}%`;
}

function sanitizeTimeseries(timeseries: AlphaTimeseries): SanitizedTimeseries {
  const { date = [], equity = [], ret_net = [] } = timeseries;
  const length = Math.max(date.length, equity.length, ret_net.length);

  const rows: ChartRow[] = [];
  let equityMin = Number.POSITIVE_INFINITY;
  let equityMax = Number.NEGATIVE_INFINITY;
  let retMin = Number.POSITIVE_INFINITY;
  let retMax = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < length; index += 1) {
    const pointDate = date[index] ?? "";
    const equityVal = sanitizeNumber(equity[index]);
    const retVal = sanitizeNumber(ret_net[index]);

    if (equityVal != null) {
      if (equityVal < equityMin) equityMin = equityVal;
      if (equityVal > equityMax) equityMax = equityVal;
    }
    if (retVal != null) {
      if (retVal < retMin) retMin = retVal;
      if (retVal > retMax) retMax = retVal;
    }

    rows.push({
      index,
      date: pointDate,
      axisLabel: buildAxisLabel(pointDate, index),
      tooltipLabel: buildTooltipLabel(pointDate, index),
      equity: equityVal,
      ret_net: retVal,
    });
  }

  const hasEquity = Number.isFinite(equityMin) && Number.isFinite(equityMax);
  const hasReturns = Number.isFinite(retMin) && Number.isFinite(retMax);

  if (!hasEquity) {
    equityMin = 0;
    equityMax = 1;
  } else if (equityMin === equityMax) {
    const pad = padDomain(equityMin);
    equityMin -= pad;
    equityMax += pad;
  }

  if (!hasReturns) {
    retMin = -1;
    retMax = 1;
  } else if (retMin === retMax) {
    const pad = padDomain(retMin);
    retMin -= pad;
    retMax += pad;
  }

  return {
    rows,
    hasEquity,
    hasReturns,
    equityDomain: [equityMin, equityMax],
    returnsDomain: [retMin, retMax],
  };
}

function EquityTooltip({ active, payload }: TooltipProps<number, string>): React.ReactElement | null {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const datum = (payload[0]?.payload as ChartRow | undefined) ?? null;
  const value = typeof payload[0]?.value === "number" ? (payload[0]?.value as number) : null;
  const tooltipLabel = datum
    ? datum.tooltipLabel || buildTooltipLabel(datum.date, datum.index)
    : "";
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__label">{tooltipLabel}</div>
      <div className="chart-tooltip__value">{formatEquity(value)}</div>
    </div>
  );
}

function ReturnsTooltip({
  active,
  payload,
}: TooltipProps<number, string>): React.ReactElement | null {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const datum = (payload[0]?.payload as ChartRow | undefined) ?? null;
  const tooltipLabel = datum
    ? datum.tooltipLabel || buildTooltipLabel(datum.date, datum.index)
    : "";
  const rawValue = payload[0]?.value;
  const value = typeof rawValue === "number" ? rawValue : null;
  const tone =
    value == null ? "chart-tooltip__value" : value >= 0 ? "chart-tooltip__value chart-tooltip__value--positive" : "chart-tooltip__value chart-tooltip__value--negative";

  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__label">{tooltipLabel}</div>
      <div className={tone}>{formatReturn(value)}</div>
    </div>
  );
}

export function TimeseriesCharts({ data, label }: TimeseriesChartsProps): React.ReactElement {
  if (!data) {
    return (
      <div className="chart-row">
        <div className="chart-card">
          <h3>Equity Curve</h3>
          <p className="chart-card__empty muted">Select an alpha to view timeseries.</p>
        </div>
        <div className="chart-card">
          <h3>Returns per Bar</h3>
          <p className="chart-card__empty muted">Select an alpha to view timeseries.</p>
        </div>
      </div>
    );
  }

  const { rows, hasEquity, hasReturns, equityDomain, returnsDomain } = useMemo(
    () => sanitizeTimeseries(data),
    [data],
  );

  const descriptor = label ? ` — ${label}` : "";

  return (
    <div className="chart-row">
      <div className="chart-card">
        <h3>Equity Curve{descriptor}</h3>
        <div className="chart-card__body">
          {hasEquity ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rows}>
                <defs>
                  <linearGradient id="equityAreaGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={COLORS.equityFillStart} />
                    <stop offset="95%" stopColor={COLORS.equityFillEnd} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  {...XAXIS_PROPS}
                  tickFormatter={(_, index) => rows[index]?.axisLabel ?? ""}
                />
                <YAxis
                  {...YAXIS_PROPS}
                  domain={equityDomain}
                  tickFormatter={(value: number) => formatEquity(value)}
                />
                <Tooltip content={<EquityTooltip />} cursor={{ stroke: COLORS.equityStroke, strokeOpacity: 0.2 }} />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke={COLORS.equityStroke}
                  fill="url(#equityAreaGradient)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <p className="chart-card__empty muted">No equity data available for this alpha.</p>
          )}
        </div>
        <div className="chart-legend">
          <span className="chart-legend__item">
            <span className="chart-legend__swatch" style={{ backgroundColor: COLORS.equityStroke }} />
            {label ?? "Equity"}
          </span>
        </div>
      </div>

      <div className="chart-card">
        <h3>Returns per Bar{descriptor}</h3>
        <div className="chart-card__body">
          {hasReturns ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows}>
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  {...XAXIS_PROPS}
                  tickFormatter={(_, index) => rows[index]?.axisLabel ?? ""}
                />
                <YAxis
                  {...YAXIS_PROPS}
                  domain={returnsDomain}
                  tickFormatter={(value: number) => formatReturn(value)}
                />
                <ReferenceLine y={0} stroke="rgba(167, 173, 179, 0.4)" strokeDasharray="4 4" />
                <Tooltip content={<ReturnsTooltip />} cursor={{ fill: "rgba(90, 180, 240, 0.08)" }} />
                <Bar dataKey="ret_net" radius={[4, 4, 0, 0]}>
                  {rows.map((item) => {
                    const value = item.ret_net;
                    let fill = COLORS.retNeutral;
                    if (value != null) {
                      if (value > 0) fill = COLORS.retPositive;
                      if (value < 0) fill = COLORS.retNegative;
                    }
                    return <Cell key={item.index} fill={fill} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="chart-card__empty muted">No return data available for this alpha.</p>
          )}
        </div>
        <div className="chart-legend">
          <span className="chart-legend__item">
            <span className="chart-legend__swatch chart-legend__swatch--positive" />
            Positive return
          </span>
          <span className="chart-legend__item">
            <span className="chart-legend__swatch chart-legend__swatch--negative" />
            Negative return
          </span>
        </div>
      </div>
    </div>
  );
}
