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
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface JobConsoleProps {
  job: PipelineJobState | null;
  connectionState?: "connected" | "retrying" | "stale" | null;
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

export function JobConsole({ job, connectionState = null, onStop, onCopyLog }: JobConsoleProps): React.ReactElement {
  if (!job) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Pipeline Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">Launch a pipeline run to monitor progress here.</p>
        </CardContent>
      </Card>
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
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="flex flex-col gap-2">
          <CardTitle className="text-xl font-bold">Pipeline Activity</CardTitle>
          {job.runName ? (
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary" className="font-mono text-xs">
                Run: {job.runName}
              </Badge>
            </div>
          ) : null}
        </div>
        <div className="flex gap-2">
          {canStop ? (
            <Button variant="destructive" size="sm" onClick={() => onStop?.(job)}>
              Stop run
            </Button>
          ) : null}
          <Button variant="outline" size="sm" onClick={handleCopy}>
            Copy log
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <ActivityStatus
          status={job.status}
          connectionState={connectionState}
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
          <div className="rounded-lg border bg-muted/50 p-4 space-y-4">
            {programMeta.length ? (
              <div className="flex flex-wrap gap-4 text-xs font-mono text-muted-foreground">
                {programMeta.map((entry) => (
                  <span key={entry.label}>
                    {entry.label}: <strong className="text-foreground">{entry.value}</strong>
                  </span>
                ))}
              </div>
            ) : null}
            <div>
              <h3 className="text-xs uppercase font-semibold text-muted-foreground mb-2">Program listing</h3>
              <pre className="font-mono text-xs overflow-x-auto whitespace-pre p-2 rounded bg-background border">
                {latestSummary.best.program ?? "Program details not available."}
              </pre>
            </div>
          </div>
        ) : (
          <div className="text-center py-6 border rounded-lg bg-muted/20">
            <h3 className="font-semibold text-sm">Awaiting first generation summary</h3>
            <p className="text-xs text-muted-foreground max-w-sm mx-auto mt-1">
              Core metrics update in real-time. Full champion analytics will appear when a generation summary is published.
            </p>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold">Live log</span>
            {job.sharpeBest !== undefined ? (
              <span className="text-muted-foreground">
                Last reported Sharpe: {job.sharpeBest === null ? "n/a" : job.sharpeBest.toFixed(4)}
              </span>
            ) : null}
          </div>
          <pre className="h-[200px] overflow-y-auto rounded-lg border bg-black text-white p-4 font-mono text-xs leading-relaxed">
            {job.log || "Waiting for log output…"}
          </pre>
        </div>
      </CardContent>
    </Card>
  );
}
