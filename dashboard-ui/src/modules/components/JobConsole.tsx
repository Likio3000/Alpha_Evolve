import React, { useMemo } from "react";
import { ActivityStatus } from "./pipeline-activity/ActivityStatus";
import { DiagnosticsPanel } from "./pipeline-activity/DiagnosticsPanel";
import { MetricEntry, MetricsDeck } from "./pipeline-activity/MetricsDeck";
import {
  buildSparklineFromSummaries,
  formatDelta,
  formatDuration,
  formatNumber,
  selectLatestSummary,
  selectPreviousSummary,
  sortEntriesByMagnitude,
  toPercentage,
} from "./pipeline-activity/utils";
import { GenerationSummary, PipelineJobState } from "../types";

interface JobConsoleProps {
  job: PipelineJobState | null;
  onStop?: (job: PipelineJobState) => void;
  onCopyLog?: (job: PipelineJobState) => void;
}

function resolvePctComplete(job: PipelineJobState, latestSummary: GenerationSummary | null): number | null {
  return toPercentage(
    (latestSummary && latestSummary.pctComplete) ??
      (job.progress?.pctComplete ??
        (job.progress && job.progress.generationsTotal
          ? job.progress.generation / Math.max(1, job.progress.generationsTotal)
          : null)),
  );
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
  const summaries = job.summaries ?? [];
  const latestSummary = selectLatestSummary(summaries);
  const previousSummary = selectPreviousSummary(summaries);

  const pctComplete = resolvePctComplete(job, latestSummary);
  const generationLabel = latestSummary
    ? `Gen ${latestSummary.generation}/${latestSummary.generationsTotal}`
    : job.progress?.generation !== undefined
      ? `Gen ${job.progress.generation}`
      : null;
  const etaLabel = formatDuration(latestSummary?.timing.etaSeconds ?? job.progress?.etaSeconds);
  const elapsedSeconds = job.progress?.elapsedSeconds ?? latestSummary?.timing.generationSeconds ?? null;

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

  const sparkline = useMemo(() => buildSparklineFromSummaries(summaries), [summaries]);

  const penaltyEntries = useMemo(
    () =>
      latestSummary
        ? sortEntriesByMagnitude(latestSummary.penalties, 5)
        : [],
    [latestSummary],
  );

  const scoreEntries = useMemo(() => {
    if (!latestSummary) {
      return [];
    }
    const filtered: Record<string, number> = {};
    for (const [key, value] of Object.entries(latestSummary.fitnessBreakdown)) {
      if (key === "result" || key === "fitness_static" || value === null) {
        continue;
      }
      const num = Number(value);
      if (Number.isFinite(num)) {
        filtered[key] = num;
      }
    }
    return sortEntriesByMagnitude(filtered, 5);
  }, [latestSummary]);

  const cadenceStats = useMemo(() => {
    const stats: Array<{ label: string; value: string }> = [];
    const generationTime = formatDuration(latestSummary?.timing.generationSeconds);
    const avgTime = formatDuration(latestSummary?.timing.averageSeconds);
    const eta = formatDuration(latestSummary?.timing.etaSeconds ?? job.progress?.etaSeconds);
    if (generationTime) {
      stats.push({ label: "Last generation", value: generationTime });
    }
    if (avgTime) {
      stats.push({ label: "Average per generation", value: avgTime });
    }
    if (eta) {
      stats.push({ label: "Pipeline ETA", value: eta });
    }
    stats.push({
      label: "Summaries received",
      value: summaries.length ? `${summaries.length}` : "0",
    });
    return stats;
  }, [job.progress?.etaSeconds, latestSummary, summaries.length]);

  const metrics: MetricEntry[] = useMemo(() => {
    const entries: MetricEntry[] = [
      {
        id: "best-fitness",
        label: "Best fitness",
        value: formatNumber(bestFitnessValue),
        delta: formatDelta(deltaFitnessValue, 4),
        trend: deltaFitnessValue === null ? null : deltaFitnessValue >= 0 ? "up" : "down",
        caption: previousSummary ? "Δ vs prior generation" : "Reported best score so far",
      },
      {
        id: "mean-ic",
        label: "Mean IC",
        value: formatNumber(meanIcValue),
        delta: formatDelta(deltaIcValue, 4),
        trend: deltaIcValue === null ? null : deltaIcValue >= 0 ? "up" : "down",
        caption: previousSummary ? "Δ vs prior generation" : "Population mean correlation",
      },
    ];

    if (latestSummary) {
      entries.push(
        {
          id: "sharpe-proxy",
          label: "Sharpe proxy",
          value: formatNumber(latestSummary.best.sharpeProxy, 3),
          caption: "Alpha-adjusted Sharpe proxy",
        },
        {
          id: "drawdown",
          label: "Drawdown",
          value: formatNumber(latestSummary.best.drawdown, 3),
          caption: "Worst peak-to-trough drawdown",
        },
      );
    } else if (job.progress?.medianFitness !== null) {
      entries.push({
        id: "median-fitness",
        label: "Median fitness",
        value: formatNumber(job.progress?.medianFitness ?? null),
        caption: "Real-time population median",
      });
    }

    if (job.progress) {
      const populationTotal = job.progress.totalIndividuals;
      const completed = job.progress.completed;
      const generation = job.progress.generation;
      entries.push({
        id: "generation-progress",
        label: "Generation progress",
        value: generation !== undefined && generation !== null ? `Gen ${generation}` : "n/a",
        caption:
          populationTotal !== undefined && populationTotal !== null
            ? `${completed}/${populationTotal} evaluated`
            : `${completed} evaluated`,
      });
    }

    return entries;
  }, [
    bestFitnessValue,
    deltaFitnessValue,
    deltaIcValue,
    job.progress,
    latestSummary,
    meanIcValue,
    previousSummary,
  ]);

  const programMeta = useMemo(() => {
    if (!latestSummary) {
      return [];
    }
    const meta: Array<{ label: string; value: string }> = [];
    const programSize = latestSummary.best.programSize;
    if (programSize !== undefined && programSize !== null) {
      meta.push({ label: "Program size", value: `${programSize}` });
    }
    const turnoverValue = latestSummary.best.turnover;
    if (turnoverValue !== undefined && turnoverValue !== null) {
      meta.push({ label: "Turnover", value: formatNumber(turnoverValue, 4) });
    }
    const populationSize = latestSummary.population.size;
    const populationUnique = latestSummary.population.uniqueFingerprints;
    if (populationSize !== undefined && populationSize !== null) {
      const uniqueLabel =
        populationUnique !== undefined && populationUnique !== null ? ` (${populationUnique} unique)` : "";
      meta.push({ label: "Population", value: `${populationSize}${uniqueLabel}` });
    }
    return meta;
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
            <button className="btn btn-warning" onClick={() => onStop?.(job)}>
              Stop run
            </button>
          ) : null}
          <button className="btn" onClick={handleCopy}>
            Copy log
          </button>
        </div>
      </div>

      <ActivityStatus
        status={job.status}
        pctComplete={pctComplete}
        generationLabel={generationLabel}
        etaLabel={etaLabel}
        lastMessage={job.lastMessage}
        lastUpdated={job.lastUpdated}
        elapsedSeconds={elapsedSeconds}
      />

      <MetricsDeck metrics={metrics} sparkline={sparkline} />

      <DiagnosticsPanel
        penalties={penaltyEntries}
        scoreContribs={scoreEntries}
        cadenceStats={cadenceStats}
        penaltiesPending={!latestSummary}
        contribsPending={!latestSummary}
      />

      {latestSummary ? (
        <div className="pipeline-activity__program-card">
          {programMeta.length ? (
            <div className="pipeline-activity__program-meta">
              {programMeta.map((entry) => (
                <span key={entry.label}>
                  {entry.label}: <strong>{entry.value}</strong>
                </span>
              ))}
            </div>
          ) : null}
          <h3 className="pipeline-activity__program-title">Program listing</h3>
          <pre className="pipeline-activity__program-snippet">
            {latestSummary.best.program ?? "Program details not available."}
          </pre>
        </div>
      ) : (
        <div className="pipeline-activity__pending">
          <h3>Awaiting first generation summary</h3>
          <p className="muted">
            Core metrics update in real-time, and full champion analytics will appear as soon as the engine publishes a
            generation summary.
          </p>
        </div>
      )}

      <div className="pipeline-activity__log pipeline-activity__log-card">
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
