import { GenerationSummary } from "../../types";

export const SPARKLINE_WIDTH = 160;
export const SPARKLINE_HEIGHT = 48;

const LABEL_OVERRIDES: Record<string, string> = {
  base_ic: "Base IC",
  sharpe_bonus: "Sharpe bonus",
  ic_std_penalty: "IC σ penalty",
  turnover_penalty: "Turnover penalty",
  parsimony_penalty: "Parsimony",
  correlation_penalty: "Correlation",
  factor_penalty: "Factor penalty",
  stress_penalty: "Stress penalty",
  drawdown_penalty: "Drawdown penalty",
  downside_penalty: "Downside penalty",
  cvar_penalty: "CVaR penalty",
  ic_tstat_bonus: "IC t-stat bonus",
};

export function formatDuration(seconds?: number | null): string | null {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) {
    return null;
  }
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}h${minutes.toString().padStart(2, "0")}m${secs.toString().padStart(2, "0")}s`;
  }
  if (minutes > 0) {
    return `${minutes}m${secs.toString().padStart(2, "0")}s`;
  }
  return `${secs}s`;
}

export function toPercentage(value?: number | null): number | null {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, Math.min(1, value)) * 100;
}

export function prettifyKey(key: string): string {
  return LABEL_OVERRIDES[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function buildSparkline(values: number[]): { points: string; min: number; max: number } | null {
  if (values.length < 2) {
    return null;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = SPARKLINE_WIDTH / (values.length - 1);
  const points = values
    .map((value, idx) => {
      const norm = (value - min) / range;
      const x = idx * stepX;
      const y = SPARKLINE_HEIGHT - norm * SPARKLINE_HEIGHT;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return { points, min, max };
}

export function formatDelta(value: number | null, digits = 4): string | null {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

export function formatNumber(value: number | null, digits = 4, fallback = "n/a"): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return fallback;
  }
  return value.toFixed(digits);
}

export function selectLatestSummary(jobSummaries: GenerationSummary[]): GenerationSummary | null {
  if (!jobSummaries || jobSummaries.length === 0) {
    return null;
  }
  return jobSummaries[jobSummaries.length - 1];
}

export function selectPreviousSummary(jobSummaries: GenerationSummary[]): GenerationSummary | null {
  if (!jobSummaries || jobSummaries.length < 2) {
    return null;
  }
  return jobSummaries[jobSummaries.length - 2];
}

export function timeAgo(timestamp?: number | null): string | null {
  if (!timestamp || !Number.isFinite(timestamp)) {
    return null;
  }
  const now = Date.now();
  const delta = Math.max(0, now - timestamp);
  const seconds = Math.round(delta / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const remMinutes = minutes % 60;
    return remMinutes ? `${hours}h ${remMinutes}m ago` : `${hours}h ago`;
  }
  const days = Math.floor(hours / 24);
  return days === 1 ? "1 day ago" : `${days} days ago`;
}

export function sortEntriesByMagnitude(
  record: Record<string, number>,
  limit: number,
  threshold = 1e-6,
): Array<{ key: string; value: number }> {
  return Object.entries(record)
    .filter(([, value]) => Number.isFinite(value) && Math.abs(value) > threshold)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, limit)
    .map(([key, value]) => ({ key, value }));
}

export function buildSparklineFromSummaries(
  summaries: GenerationSummary[],
  take = 60,
): { points: string; min: number; max: number } | null {
  if (!summaries.length) {
    return null;
  }
  const values = summaries
    .map((entry) => entry.best.fitness)
    .filter((value) => Number.isFinite(value))
    .slice(-take);
  if (!values.length) {
    return null;
  }
  return buildSparkline(values);
}
