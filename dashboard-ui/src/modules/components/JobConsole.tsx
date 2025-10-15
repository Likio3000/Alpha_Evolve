import React, { useMemo } from "react";
import { GenerationSummary, PipelineJobState } from "../types";

interface JobConsoleProps {
  job: PipelineJobState | null;
  onStop?: (job: PipelineJobState) => void;
  onCopyLog?: (job: PipelineJobState) => void;
}

const SPARKLINE_WIDTH = 160;
const SPARKLINE_HEIGHT = 48;

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

function formatDuration(seconds?: number | null): string | null {
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

function toPercentage(value?: number | null): number | null {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, Math.min(1, value)) * 100;
}

function prettifyKey(key: string): string {
  return LABEL_OVERRIDES[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildSparkline(values: number[]): { points: string; min: number; max: number } | null {
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

function formatDelta(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "";
  }
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(4)}`;
}

function selectLatestSummary(job: PipelineJobState): GenerationSummary | null {
  if (!job.summaries || job.summaries.length === 0) {
    return null;
  }
  return job.summaries[job.summaries.length - 1];
}

function selectPreviousSummary(job: PipelineJobState): GenerationSummary | null {
  if (!job.summaries || job.summaries.length < 2) {
    return null;
  }
  return job.summaries[job.summaries.length - 2];
}

export function JobConsole({ job, onStop, onCopyLog }: JobConsoleProps): React.ReactElement {
  if (!job) {
    return (
      <section className="panel pipeline-activity">
        <h2>Pipeline Activity</h2>
        <p className="muted">Launch a pipeline run to monitor progress here.</p>
      </section>
    );
  }

  const canStop = job.status === "running" && onStop;
  const latestSummary = selectLatestSummary(job);
  const previousSummary = selectPreviousSummary(job);

  const pctComplete = toPercentage(
    (latestSummary && latestSummary.pctComplete) ??
      (job.progress?.pctComplete ??
        (job.progress && job.progress.generationsTotal
          ? job.progress.generation / Math.max(1, job.progress.generationsTotal)
          : null)),
  );
  const generationLabel = latestSummary
    ? `Gen ${latestSummary.generation}/${latestSummary.generationsTotal}`
    : job.progress?.generation !== undefined
      ? `Gen ${job.progress.generation}`
      : null;
  const etaLabel = formatDuration(latestSummary?.timing.etaSeconds ?? job.progress?.etaSeconds);
  const bestFitness = latestSummary?.best.fitness ?? job.progress?.bestFitness ?? null;
  const meanIc = latestSummary?.best.meanIc ?? job.progress?.medianFitness ?? null;
  const deltaFitness =
    latestSummary && previousSummary ? latestSummary.best.fitness - previousSummary.best.fitness : null;
  const deltaIc =
    latestSummary && previousSummary ? latestSummary.best.meanIc - previousSummary.best.meanIc : null;
  const bestFitnessValue = typeof bestFitness === "number" && Number.isFinite(bestFitness) ? bestFitness : null;
  const meanIcValue = typeof meanIc === "number" && Number.isFinite(meanIc) ? meanIc : null;
  const deltaFitnessValue =
    typeof deltaFitness === "number" && Number.isFinite(deltaFitness) ? deltaFitness : null;
  const deltaIcValue = typeof deltaIc === "number" && Number.isFinite(deltaIc) ? deltaIc : null;

  const summaries = job.summaries ?? [];
  const sparkline = useMemo(() => {
    if (!summaries.length) {
      return null;
    }
    const values = summaries
      .map((entry) => entry.best.fitness)
      .filter((value) => Number.isFinite(value))
      .slice(-60);
    return buildSparkline(values);
  }, [summaries]);

  const penaltyEntries = useMemo(() => {
    if (!latestSummary) {
      return [];
    }
    return Object.entries(latestSummary.penalties)
      .filter(([, value]) => Number.isFinite(value) && Math.abs(Number(value)) > 1e-6)
      .sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])))
      .slice(0, 5);
  }, [latestSummary]);

  const scoreEntries = useMemo(() => {
    if (!latestSummary) {
      return [];
    }
    return Object.entries(latestSummary.fitnessBreakdown)
      .filter(([key, value]) => key !== "result" && key !== "fitness_static" && value !== null && Number.isFinite(value))
      .sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])))
      .slice(0, 5);
  }, [latestSummary]);

  const handleCopy = () => {
    if (!onCopyLog) return;
    onCopyLog(job);
  };

  return (
    <section className="panel pipeline-activity">
      <div className="panel-header">
        <h2>Pipeline Activity</h2>
        <div className="panel-actions">
          {canStop ? (
            <button className="btn btn-warning" onClick={() => onStop(job)}>
              Stop run
            </button>
          ) : null}
          <button className="btn" onClick={handleCopy}>
            Copy log
          </button>
        </div>
      </div>
      <div className="pipeline-activity__status-row">
        <div className={`status-badge status-badge--${job.status}`}>{job.status}</div>
        {generationLabel ? <span className="pipeline-activity__status-meta">{generationLabel}</span> : null}
        {etaLabel ? <span className="pipeline-activity__status-meta muted">ETA {etaLabel}</span> : null}
      </div>

      <div className="pipeline-activity__progress">
        <div className="pipeline-activity__progress-bar">
          <div
            className="pipeline-activity__progress-bar-fill"
            style={{ width: `${pctComplete !== null ? pctComplete : 0}%` }}
          />
        </div>
        <div className="pipeline-activity__progress-meta">
          <span>
            {pctComplete !== null ? `${pctComplete.toFixed(1)}% complete` : "Waiting for progress updates…"}
          </span>
          {job.lastMessage ? <span className="muted">{job.lastMessage}</span> : null}
        </div>
      </div>

      {latestSummary ? (
        <>
          <div className="pipeline-activity__metrics">
            <div className="pipeline-activity__metric">
              <span className="pipeline-activity__metric-label">Best fitness</span>
              <span className="pipeline-activity__metric-value">
                {bestFitnessValue !== null ? bestFitnessValue.toFixed(4) : "n/a"}
              </span>
              {deltaFitnessValue !== null ? (
                <span
                  className={`metric-delta ${
                    deltaFitnessValue >= 0 ? "metric-delta--up" : "metric-delta--down"
                  }`}
                >
                  {formatDelta(deltaFitnessValue)}
                </span>
              ) : null}
            </div>
            <div className="pipeline-activity__metric">
              <span className="pipeline-activity__metric-label">Mean IC</span>
              <span className="pipeline-activity__metric-value">
                {meanIcValue !== null ? meanIcValue.toFixed(4) : "n/a"}
              </span>
              {deltaIcValue !== null ? (
                <span
                  className={`metric-delta ${deltaIcValue >= 0 ? "metric-delta--up" : "metric-delta--down"}`}
                >
                  {formatDelta(deltaIcValue)}
                </span>
              ) : null}
            </div>
            <div className="pipeline-activity__metric">
              <span className="pipeline-activity__metric-label">Turnover</span>
              <span className="pipeline-activity__metric-value">
                {latestSummary.best.turnover.toFixed(4)}
              </span>
            </div>
            <div className="pipeline-activity__metric pipeline-activity__metric--sparkline">
              <span className="pipeline-activity__metric-label">Fitness trend</span>
              {sparkline ? (
                <svg
                  className="pipeline-activity__sparkline"
                  viewBox={`0 0 ${SPARKLINE_WIDTH} ${SPARKLINE_HEIGHT}`}
                  xmlns="http://www.w3.org/2000/svg"
                  preserveAspectRatio="none"
                >
                  <polyline points={sparkline.points} />
                </svg>
              ) : (
                <span className="muted">Collecting data…</span>
              )}
            </div>
          </div>

          <div className="pipeline-activity__detail-grid">
            <div>
              <h3>Penalty breakdown</h3>
              {penaltyEntries.length ? (
                <ul className="pipeline-activity__list">
                  {penaltyEntries.map(([key, value]) => (
                    <li key={key}>
                      <span>{prettifyKey(key)}</span>
                      <span className="pipeline-activity__penalty-value">
                        -{Math.abs(Number(value)).toFixed(4)}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No active penalties this generation.</p>
              )}
            </div>

            <div>
              <h3>Score contributions</h3>
              {scoreEntries.length ? (
                <ul className="pipeline-activity__list">
                  {scoreEntries.map(([key, value]) => (
                    <li key={key}>
                      <span>{prettifyKey(key)}</span>
                      <span
                        className={`pipeline-activity__contrib ${
                          Number(value) >= 0 ? "pipeline-activity__contrib--pos" : "pipeline-activity__contrib--neg"
                        }`}
                      >
                        {formatDelta(Number(value))}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">Awaiting breakdown data…</p>
              )}
            </div>

            <div>
              <h3>Champion fingerprint</h3>
              <div className="pipeline-activity__program">
                <code>{latestSummary.best.fingerprint ?? "n/a"}</code>
              </div>
              <h4>Program</h4>
              <pre className="pipeline-activity__program-snippet">{latestSummary.best.program}</pre>
              <p className="muted">
                Population: {latestSummary.population.size} candidates ({latestSummary.population.uniqueFingerprints} unique)
              </p>
            </div>
          </div>
        </>
      ) : (
        <p className="muted">Waiting for first generation summary…</p>
      )}

      <div className="pipeline-activity__log">
        <div className="pipeline-activity__log-header">
          <span>Live log</span>
          {job.sharpeBest !== undefined ? (
            <span className="muted">
              Last reported Sharpe: {job.sharpeBest === null ? "n/a" : job.sharpeBest.toFixed(4)}
            </span>
          ) : null}
        </div>
        <pre className="log-viewer">{job.log || "Waiting for log output…"}</pre>
      </div>
    </section>
  );
}
