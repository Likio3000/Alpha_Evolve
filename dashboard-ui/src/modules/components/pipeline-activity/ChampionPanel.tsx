import React from "react";
import { prettifyKey } from "./utils";

interface ExposureEntry {
  key: string;
  value: number;
}

interface ChampionPanelProps {
  fingerprint?: string | null;
  program?: string | null;
  populationSize?: number | null;
  populationUnique?: number | null;
  programSize?: number | null;
  turnover?: number | null;
  factorExposures: ExposureEntry[];
  regimeExposures: ExposureEntry[];
  stressMetrics: ExposureEntry[];
  pending?: boolean;
}

export function ChampionPanel({
  fingerprint,
  program,
  populationSize,
  populationUnique,
  programSize,
  turnover,
  factorExposures,
  regimeExposures,
  stressMetrics,
  pending = false,
}: ChampionPanelProps): React.ReactElement {
  const populationLabel =
    populationSize !== undefined && populationSize !== null
      ? `${populationSize}`
      : pending
        ? "…"
        : "n/a";
  const populationUniqueLabel =
    populationUnique !== undefined && populationUnique !== null ? `${populationUnique}` : "n/a";
  const programLabel = program ?? (pending ? "Program details will appear here once available." : "n/a");

  return (
    <div className="pipeline-activity__champion-grid">
      <div className="pipeline-activity__champion-card">
        <h3>Champion fingerprint</h3>
        <div className="pipeline-activity__program">
          <code>{fingerprint ?? (pending ? "Pending first summary…" : "n/a")}</code>
        </div>
        <div className="pipeline-activity__champion-meta">
          <span>
            Program size: <strong>{programSize ?? (pending ? "…" : "n/a")}</strong>
          </span>
          <span>
            Turnover:{" "}
            <strong>
              {turnover !== undefined && turnover !== null
                ? turnover.toFixed(4)
                : pending
                  ? "…"
                  : "n/a"}
            </strong>
          </span>
          <span>
            Population: <strong>{populationLabel}</strong> ({populationUniqueLabel} unique)
          </span>
        </div>
      </div>

      <div className="pipeline-activity__champion-card pipeline-activity__champion-card--code">
        <h3>Program listing</h3>
        <pre className="pipeline-activity__program-snippet">{programLabel}</pre>
      </div>

      <div className="pipeline-activity__champion-card">
        <h3>Top factor exposures</h3>
        {factorExposures.length ? (
          <ul className="pipeline-activity__stat-list">
            {factorExposures.map(({ key, value }) => (
              <li key={key}>
                <span className="muted">{prettifyKey(key)}</span>
                <span
                  className={`pipeline-activity__contrib ${
                    value >= 0 ? "pipeline-activity__contrib--pos" : "pipeline-activity__contrib--neg"
                  }`}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">{pending ? "Awaiting exposure analytics…" : "No material exposures reported."}</p>
        )}
      </div>

      <div className="pipeline-activity__champion-card">
        <h3>Regime & stress signals</h3>
        <div className="pipeline-activity__exposure-columns">
          <div>
            <h4>Regime</h4>
            {regimeExposures.length ? (
              <ul className="pipeline-activity__stat-list">
                {regimeExposures.map(({ key, value }) => (
                  <li key={key}>
                    <span className="muted">{prettifyKey(key)}</span>
                    <span
                      className={`pipeline-activity__contrib ${
                        value >= 0 ? "pipeline-activity__contrib--pos" : "pipeline-activity__contrib--neg"
                      }`}
                    >
                      {value >= 0 ? "+" : ""}
                      {value.toFixed(3)}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">{pending ? "Regime analytics pending…" : "No regime bias detected."}</p>
            )}
          </div>
          <div>
            <h4>Stress</h4>
            {stressMetrics.length ? (
              <ul className="pipeline-activity__stat-list">
                {stressMetrics.map(({ key, value }) => (
                  <li key={key}>
                    <span className="muted">{prettifyKey(key)}</span>
                    <span>{value.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">{pending ? "Awaiting stress metrics…" : "No stress metrics reported."}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
