import React from "react";
import { prettifyKey } from "./utils";

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
    <div className="pipeline-activity__insights">
      <div className="pipeline-activity__insight-card">
        <h3>Penalty breakdown</h3>
        {penalties.length ? (
          <ul className="pipeline-activity__list">
            {penalties.map(({ key, value }) => (
              <li key={key}>
                <span>{prettifyKey(key)}</span>
                <span className="pipeline-activity__penalty-value">-{Math.abs(value).toFixed(4)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">
            {penaltiesPending ? "Waiting for penalty data…" : "No active penalties this generation."}
          </p>
        )}
      </div>

      <div className="pipeline-activity__insight-card">
        <h3>Score contributions</h3>
        {scoreContribs.length ? (
          <ul className="pipeline-activity__list">
            {scoreContribs.map(({ key, value }) => (
              <li key={key}>
                <span>{prettifyKey(key)}</span>
                <span
                  className={`pipeline-activity__contrib ${
                    value >= 0 ? "pipeline-activity__contrib--pos" : "pipeline-activity__contrib--neg"
                  }`}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(4)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">
            {contribsPending ? "Awaiting fitness breakdowns…" : "No contribution data available."}
          </p>
        )}
      </div>

      <div className="pipeline-activity__insight-card">
        <h3>Run cadence</h3>
        <ul className="pipeline-activity__stat-list">
          {cadenceStats.map((stat) => (
            <li key={stat.label}>
              <span className="muted">{stat.label}</span>
              <span>{stat.value}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
