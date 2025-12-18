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
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

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

// Keeping consistent theme colors for charts, but could use CSS variables if Recharts supported them easily.
const COLORS = {
  equityStroke: "#3b82f6", // blue-500
  equityFillStart: "rgba(59, 130, 246, 0.5)",
  equityFillEnd: "rgba(59, 130, 246, 0.05)",
  grid: "rgba(255, 255, 255, 0.1)",
  axis: "#94a3b8", // slate-400
  tooltipBg: "rgba(15, 23, 42, 0.95)", // slate-950
  tooltipBorder: "rgba(59, 130, 246, 0.5)",
  retPositive: "#10b981", // emerald-500
  retNegative: "#ef4444", // red-500
  retNeutral: "#eab308", // yellow-500
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
    <div className="bg-slate-950 border border-blue-500/50 rounded-lg p-2 shadow-xl backdrop-blur-md">
      <div className="text-[10px] text-slate-400 font-medium mb-1 uppercase tracking-wider">{tooltipLabel}</div>
      <div className="text-sm font-bold text-white font-mono">{formatEquity(value)}</div>
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

  let toneClass = "text-white";
  if (value != null) {
    if (value > 0) toneClass = "text-emerald-400";
    else if (value < 0) toneClass = "text-red-400";
  }

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg p-2 shadow-xl backdrop-blur-md">
      <div className="text-[10px] text-slate-400 font-medium mb-1 uppercase tracking-wider">{tooltipLabel}</div>
      <div className={cn("text-sm font-bold font-mono", toneClass)}>{formatReturn(value)}</div>
    </div>
  );
}

export function TimeseriesCharts({ data, label }: TimeseriesChartsProps): React.ReactElement {
  if (!data) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="h-[300px] flex items-center justify-center text-muted-foreground bg-muted/20">
          <div className="text-center">
            <h3 className="text-sm font-semibold mb-1">Equity Curve</h3>
            <p className="text-xs">Select an alpha to view timeseries.</p>
          </div>
        </Card>
        <Card className="h-[300px] flex items-center justify-center text-muted-foreground bg-muted/20">
          <div className="text-center">
            <h3 className="text-sm font-semibold mb-1">Returns per Bar</h3>
            <p className="text-xs">Select an alpha to view timeseries.</p>
          </div>
        </Card>
      </div>
    );
  }

  const { rows, hasEquity, hasReturns, equityDomain, returnsDomain } = useMemo(
    () => sanitizeTimeseries(data),
    [data],
  );

  const descriptor = label ? ` — ${label}` : "";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card className="flex flex-col h-[300px]">
        <CardHeader className="py-3 px-4 pb-0">
          <CardTitle className="text-sm font-medium text-muted-foreground">Equity Curve{descriptor}</CardTitle>
        </CardHeader>
        <CardContent className="flex-grow min-h-0 p-4">
          {hasEquity ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rows}>
                <defs>
                  <linearGradient id="equityAreaGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={COLORS.equityFillStart} />
                    <stop offset="95%" stopColor={COLORS.equityFillEnd} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="date"
                  {...XAXIS_PROPS}
                  tickFormatter={(_, index) => rows[index]?.axisLabel ?? ""}
                  height={20}
                />
                <YAxis
                  {...YAXIS_PROPS}
                  domain={equityDomain}
                  tickFormatter={(value: number) => formatEquity(value)}
                  width={40}
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
            <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
              No equity data available.
            </div>
          )}
        </CardContent>
        <div className="px-4 pb-3 flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-sm bg-blue-500" />
          <span className="text-xs text-muted-foreground">{label ?? "Equity"}</span>
        </div>
      </Card>

      <Card className="flex flex-col h-[300px]">
        <CardHeader className="py-3 px-4 pb-0">
          <CardTitle className="text-sm font-medium text-muted-foreground">Returns per Bar{descriptor}</CardTitle>
        </CardHeader>
        <CardContent className="flex-grow min-h-0 p-4">
          {hasReturns ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows}>
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="date"
                  {...XAXIS_PROPS}
                  tickFormatter={(_, index) => rows[index]?.axisLabel ?? ""}
                  height={20}
                />
                <YAxis
                  {...YAXIS_PROPS}
                  domain={returnsDomain}
                  tickFormatter={(value: number) => formatReturn(value)}
                  width={40}
                />
                <ReferenceLine y={0} stroke="rgba(255, 255, 255, 0.2)" strokeDasharray="4 4" />
                <Tooltip content={<ReturnsTooltip />} cursor={{ fill: "rgba(255, 255, 255, 0.03)" }} />
                <Bar dataKey="ret_net" radius={[2, 2, 0, 0]}>
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
            <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
              No return data available.
            </div>
          )}
        </CardContent>
        <div className="px-4 pb-3 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
            <span className="text-xs text-muted-foreground">Positive</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-sm bg-red-500" />
            <span className="text-xs text-muted-foreground">Negative</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
