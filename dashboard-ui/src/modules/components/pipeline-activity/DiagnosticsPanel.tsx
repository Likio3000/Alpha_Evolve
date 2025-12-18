import React from "react";
import { prettifyKey } from "./utils";
import { cn } from "@/lib/utils";

interface DiagnosticsPanelProps {
  penalties: Array<{ key: string; value: number }>;
  scoreContribs: Array<{ key: string; value: number }>;
  cadenceStats: Array<{ label: string; value: string }>;
  penaltiesPending?: boolean;
  contribsPending?: boolean;
}

export function DiagnosticsPanel({
  penalties,
  scoreContribs,
  cadenceStats,
  penaltiesPending = false,
  contribsPending = false,
}: DiagnosticsPanelProps): React.ReactElement {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="space-y-3">
        <h3 className="text-sm font-semibold border-b pb-1">Penalty breakdown</h3>
        {penalties.length ? (
          <ul className="space-y-1 text-sm">
            {penalties.map(({ key, value }) => (
              <li key={key} className="flex justify-between items-center group">
                <span className="text-muted-foreground group-hover:text-foreground transition-colors">{prettifyKey(key)}</span>
                <span className="font-mono text-destructive">-{Math.abs(value).toFixed(4)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-muted-foreground italic">
            {penaltiesPending ? "Waiting for penalty data…" : "No active penalties this generation."}
          </p>
        )}
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold border-b pb-1">Score contributions</h3>
        {scoreContribs.length ? (
          <ul className="space-y-1 text-sm">
            {scoreContribs.map(({ key, value }) => (
              <li key={key} className="flex justify-between items-center group">
                <span className="text-muted-foreground group-hover:text-foreground transition-colors">{prettifyKey(key)}</span>
                <span
                  className={cn(
                    "font-mono",
                    value >= 0 ? "text-green-600 dark:text-green-400" : "text-destructive"
                  )}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(4)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-muted-foreground italic">
            {contribsPending ? "Awaiting fitness breakdowns…" : "No contribution data available."}
          </p>
        )}
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold border-b pb-1">Run cadence</h3>
        <ul className="space-y-2 text-sm">
          {cadenceStats.map((stat) => (
            <li key={stat.label} className="flex justify-between items-center bg-muted/30 p-2 rounded">
              <span className="text-xs text-muted-foreground">{stat.label}</span>
              <span className="font-mono">{stat.value}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
