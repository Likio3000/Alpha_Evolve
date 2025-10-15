import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchAlphaTimeseries,
  fetchBacktestSummary,
  fetchJobActivity,
  fetchLastRun,
  fetchRuns,
  startPipelineRun,
  stopPipelineJob,
  updateRunLabel,
} from "./api";
import { PipelineControls } from "./components/PipelineControls";
import { RunList } from "./components/RunList";
import { BacktestTable } from "./components/BacktestTable";
import { TimeseriesCharts } from "./components/TimeseriesCharts";
import { JobConsole } from "./components/JobConsole";
import { HeaderNav } from "./components/HeaderNav";
import { RunnerCanvas } from "./components/RunnerCanvas";
import { SettingsPanel } from "./components/SettingsPanel";
import { usePolling } from "./hooks/usePolling";
import {
  AlphaTimeseries,
  BacktestRow,
  GenerationProgressState,
  GenerationSummary,
  PipelineJobState,
  PipelineRunRequest,
  RunSummary,
} from "./types";

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function extractSharpeFromLog(log: string | null | undefined): number | null {
  if (!log) {
    return null;
  }
  const matches = [...log.matchAll(/Sharpe\(best\)\s*=\s*([+\-]?[0-9.]+)/g)];
  if (!matches.length) {
    return null;
  }
  const last = matches[matches.length - 1];
  const value = Number(last[1]);
  return Number.isFinite(value) ? value : null;
}

const SUMMARY_HISTORY_LIMIT = 400;

function toNumber(value: unknown, fallback = 0): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function toNullableNumber(value: unknown): number | null {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function mapNumberRecord(source: unknown): Record<string, number> {
  const record = asRecord(source);
  if (!record) {
    return {};
  }
  const result: Record<string, number> = {};
  for (const [key, val] of Object.entries(record)) {
    const num = Number(val);
    if (Number.isFinite(num)) {
      result[key] = num;
    }
  }
  return result;
}

function mapNestedNumberRecord(source: unknown): Record<string, Record<string, number>> {
  const record = asRecord(source);
  if (!record) {
    return {};
  }
  const result: Record<string, Record<string, number>> = {};
  for (const [key, val] of Object.entries(record)) {
    result[key] = mapNumberRecord(val);
  }
  return result;
}

function mapGenerationProgress(raw: unknown): GenerationProgressState | null {
  const record = asRecord(raw);
  if (!record) {
    return null;
  }
  const generation = toNumber(record.gen ?? record.generation, NaN);
  const completed = toNumber(record.completed, NaN);
  if (!Number.isFinite(generation) || !Number.isFinite(completed)) {
    return null;
  }
  const total = toNullableNumber(record.total ?? record.population ?? record.total_individuals);
  const pctComplete = toNullableNumber(record.pct_complete ?? record.pct);
  let derivedPct = pctComplete;
  if (derivedPct === null && total && total > 0) {
    derivedPct = Math.min(1, generation / total);
  }
  return {
    generation,
    generationsTotal: toNullableNumber(record.generations_total ?? record.total_gens),
    pctComplete: derivedPct,
    completed,
    totalIndividuals: total,
    bestFitness: toNullableNumber(record.best),
    medianFitness: toNullableNumber(record.median),
    elapsedSeconds: toNullableNumber(record.elapsed_sec ?? record.elapsed),
    etaSeconds: toNullableNumber(record.eta_sec ?? record.eta),
  };
}

function mapGenerationSummary(raw: unknown): GenerationSummary | null {
  const record = asRecord(raw);
  if (!record) {
    return null;
  }
  const generation = toNumber(record.generation, NaN);
  const total = toNumber(record.generations_total, NaN);
  if (!Number.isFinite(generation) || !Number.isFinite(total)) {
    return null;
  }
  const pctCompleteRaw = Number(record.pct_complete);
  const pctComplete = Number.isFinite(pctCompleteRaw)
    ? pctCompleteRaw
    : generation / Math.max(1, total);

  const bestRecord = asRecord(record.best);
  if (!bestRecord) {
    return null;
  }

  const timingRecord = asRecord(record.timing);
  const populationRecord = asRecord(record.population);
  const penalties = mapNumberRecord(record.penalties);
  const fitnessBreakdownRaw = asRecord(record.fitness_breakdown);
  const fitnessBreakdown: Record<string, number | null> = {};
  if (fitnessBreakdownRaw) {
    for (const [key, val] of Object.entries(fitnessBreakdownRaw)) {
      const num = Number(val);
      fitnessBreakdown[key] = Number.isFinite(num) ? num : null;
    }
  }

  const best: GenerationSummary["best"] = {
    fitness: toNumber(bestRecord.fitness),
    fitnessStatic: toNullableNumber(bestRecord.fitness_static),
    meanIc: toNumber(bestRecord.mean_ic),
    icStd: toNumber(bestRecord.ic_std),
    turnover: toNumber(bestRecord.turnover),
    sharpeProxy: toNumber(bestRecord.sharpe_proxy),
    sortino: toNumber(bestRecord.sortino),
    drawdown: toNumber(bestRecord.drawdown),
    downsideDeviation: toNumber(bestRecord.downside_deviation),
    cvar: toNumber(bestRecord.cvar),
    factorPenalty: toNumber(bestRecord.factor_penalty),
    fingerprint: typeof bestRecord.fingerprint === "string" ? bestRecord.fingerprint : null,
    programSize: Math.trunc(toNumber(bestRecord.program_size)),
    program: typeof bestRecord.program === "string" ? bestRecord.program : "",
    horizonMetrics: mapNestedNumberRecord(bestRecord.horizon_metrics),
    factorExposures: mapNumberRecord(bestRecord.factor_exposures),
    regimeExposures: mapNumberRecord(bestRecord.regime_exposures),
    transactionCosts: mapNumberRecord(bestRecord.transaction_costs),
    stressMetrics: mapNumberRecord(bestRecord.stress_metrics),
  };

  const timing = {
    generationSeconds: toNumber(timingRecord?.generation_seconds),
    averageSeconds: toNullableNumber(timingRecord?.average_seconds),
    etaSeconds: toNullableNumber(timingRecord?.eta_seconds),
  };

  const population = {
    size: Math.trunc(toNumber(populationRecord?.size)),
    uniqueFingerprints: Math.trunc(toNumber(populationRecord?.unique_fingerprints)),
  };

  return {
    generation,
    generationsTotal: total,
    pctComplete,
    best,
    penalties,
    fitnessBreakdown,
    timing,
    population,
  };
}

export function App(): React.ReactElement {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [lastRunPath, setLastRunPath] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);

  const [selectedRunPath, setSelectedRunPath] = useState<string | null>(null);

  const [backtestRows, setBacktestRows] = useState<BacktestRow[]>([]);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);

  const [selectedRow, setSelectedRow] = useState<BacktestRow | null>(null);
  const [alphaTimeseries, setAlphaTimeseries] = useState<AlphaTimeseries | null>(null);
  const [timeseriesLoading, setTimeseriesLoading] = useState(false);
  const [timeseriesError, setTimeseriesError] = useState<string | null>(null);

  const [job, setJob] = useState<PipelineJobState | null>(null);
  const [banner, setBanner] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "settings">("overview");
  const closeEventStream = useCallback(() => {}, []);
  const [pendingRunSelectionBaseline, setPendingRunSelectionBaseline] = useState<string | null | undefined>(undefined);

  const selectedRun = useMemo(() => {
    if (!selectedRunPath) {
      return null;
    }
    return runs.find((run) => run.path === selectedRunPath) ?? null;
  }, [runs, selectedRunPath]);

  const selectedAlphaKey = useMemo(() => {
    if (!selectedRow) return null;
    return selectedRow.TimeseriesFile || selectedRow.TS || selectedRow.AlphaID || null;
  }, [selectedRow]);

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const [runListResult, lastRunResult] = await Promise.allSettled([fetchRuns(), fetchLastRun()]);
      if (runListResult.status !== "fulfilled") {
        throw runListResult.reason;
      }
      const runItems = runListResult.value;
      setRuns(runItems);
      setRunsError(null);
      setLastRunPath(
        lastRunResult.status === "fulfilled" ? lastRunResult.value.run_dir ?? null : null,
      );
    } catch (error) {
      setRunsError(formatError(error));
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  useEffect(() => {
    if (!runs.length) {
      if (selectedRunPath !== null) {
        setSelectedRunPath(null);
      }
      if (pendingRunSelectionBaseline !== undefined) {
        setPendingRunSelectionBaseline(undefined);
      }
      return;
    }

    const latestPath = runs[0]?.path ?? null;

    if (pendingRunSelectionBaseline !== undefined) {
      if (latestPath && latestPath !== pendingRunSelectionBaseline) {
        if (selectedRunPath !== latestPath) {
          setSelectedRunPath(latestPath);
        }
        setPendingRunSelectionBaseline(undefined);
        return;
      }
      if (lastRunPath && lastRunPath !== pendingRunSelectionBaseline) {
        const match = runs.find((run) => run.path === lastRunPath);
        if (match && match.path !== selectedRunPath) {
          setSelectedRunPath(match.path);
        }
        setPendingRunSelectionBaseline(undefined);
        return;
      }
    }

    if (selectedRunPath && runs.some((run) => run.path === selectedRunPath)) {
      return;
    }

    if (lastRunPath) {
      const match = runs.find((run) => run.path === lastRunPath);
      if (match && match.path !== selectedRunPath) {
        setSelectedRunPath(match.path);
        return;
      }
    }

    if (latestPath !== selectedRunPath) {
      setSelectedRunPath(latestPath);
    }
  }, [runs, selectedRunPath, lastRunPath, pendingRunSelectionBaseline]);

  useEffect(() => {
    if (!selectedRunPath) {
      setBacktestRows([]);
      setSelectedRow(null);
      setAlphaTimeseries(null);
      return;
    }

    let cancelled = false;
    setBacktestLoading(true);
    setBacktestError(null);
    setSelectedRow(null);

    (async () => {
      try {
        const rows = await fetchBacktestSummary(selectedRunPath);
        if (cancelled) return;
        setBacktestRows(rows);
        setSelectedRow(rows[0] ?? null);
      } catch (error) {
        if (cancelled) return;
        setBacktestRows([]);
        setSelectedRow(null);
        setAlphaTimeseries(null);
        setBacktestError(formatError(error));
      } finally {
        if (!cancelled) {
          setBacktestLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedRunPath]);

  useEffect(() => {
    if (!selectedRunPath || !selectedRow) {
      setAlphaTimeseries(null);
      setTimeseriesError(null);
      return;
    }

    let cancelled = false;
    setTimeseriesLoading(true);
    setTimeseriesError(null);

    (async () => {
      try {
        const data = await fetchAlphaTimeseries(
          selectedRunPath,
          selectedRow.AlphaID ?? undefined,
          selectedRow.TimeseriesFile ?? undefined,
        );
        if (cancelled) return;
        setAlphaTimeseries(data);
      } catch (error) {
        if (cancelled) return;
        setTimeseriesError(formatError(error));
        setAlphaTimeseries(null);
      } finally {
        if (!cancelled) {
          setTimeseriesLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedRunPath, selectedRow]);

  const handleSelectRun = useCallback((run: RunSummary) => {
    setSelectedRunPath(run.path);
  }, []);

  const handleSelectRow = useCallback((row: BacktestRow) => {
    setSelectedRow(row);
  }, []);

  const refreshJob = useCallback(
    async (jobIdOverride?: string) => {
      const activeJobId = jobIdOverride ?? job?.jobId;
      if (!activeJobId) {
        return;
      }
      try {
        const snapshot = await fetchJobActivity(activeJobId);
        const summaryPayload = Array.isArray(snapshot.summaries) ? snapshot.summaries : [];
        const mappedSummaries = summaryPayload
          .map((entry) => mapGenerationSummary(entry))
          .filter((entry): entry is GenerationSummary => Boolean(entry));
        const trimmedSummaries =
          mappedSummaries.length > SUMMARY_HISTORY_LIMIT
            ? mappedSummaries.slice(mappedSummaries.length - SUMMARY_HISTORY_LIMIT)
            : mappedSummaries;
        let progressState = mapGenerationProgress(snapshot.progress);
        if (!progressState && trimmedSummaries.length) {
          const latestSummary = trimmedSummaries[trimmedSummaries.length - 1];
          progressState = {
            generation: latestSummary.generation,
            generationsTotal: latestSummary.generationsTotal,
            pctComplete: latestSummary.pctComplete,
            completed: latestSummary.population.size,
            totalIndividuals: latestSummary.population.size,
            bestFitness: latestSummary.best.fitness,
            medianFitness: latestSummary.best.meanIc,
            elapsedSeconds: latestSummary.timing.generationSeconds,
            etaSeconds: latestSummary.timing.etaSeconds,
          };
        }
        const updatedAtMs = (() => {
          const raw = snapshot.updated_at;
          const numeric = typeof raw === "number" ? raw : Number(raw);
          if (Number.isFinite(numeric)) {
            return Math.max(Date.now(), Math.trunc(numeric > 1e12 ? numeric : numeric * 1000));
          }
          return Date.now();
        })();

        setJob((current) => {
          const base: PipelineJobState =
            current && current.jobId === activeJobId
              ? current
              : {
                  jobId: activeJobId,
                  status: "running",
                  lastMessage: "Pipeline running…",
                  lastUpdated: Date.now(),
                  log: "",
                  sharpeBest: null,
                  progress: null,
                  summaries: [],
                };
          const next: PipelineJobState = {
            ...base,
            lastUpdated: updatedAtMs,
          };
          if (!snapshot.exists) {
            next.status = "error";
            next.lastMessage = "Job no longer exists.";
            return next;
          }
          const statusRaw = snapshot.status;
          if (typeof statusRaw === "string" && ["idle", "running", "error", "complete"].includes(statusRaw)) {
            next.status = statusRaw as PipelineJobState["status"];
          } else if (snapshot.running) {
            next.status = "running";
          } else if (next.status === "running") {
            next.status = "complete";
            if (!next.lastMessage) {
              next.lastMessage = "Pipeline finished.";
            }
          }
          const messageRaw = snapshot.last_message;
          if (typeof messageRaw === "string" && messageRaw.trim()) {
            next.lastMessage = messageRaw;
          } else if (!next.lastMessage && next.status === "running") {
            next.lastMessage = "Pipeline running…";
          }
          if (typeof snapshot.log === "string") {
            const logText = snapshot.log;
            next.log = logText;
            const sharpeFromLog = extractSharpeFromLog(logText);
            if (sharpeFromLog !== null) {
              next.sharpeBest = sharpeFromLog;
            }
          }
          const sharpeRaw = snapshot.sharpe_best;
          if (sharpeRaw !== undefined && sharpeRaw !== null) {
            const value = Number(sharpeRaw);
            if (Number.isFinite(value)) {
              next.sharpeBest = value;
            }
          }
          if (progressState) {
            next.progress = progressState;
          }
          if (trimmedSummaries.length) {
            next.summaries = trimmedSummaries;
          }
          const logPath = snapshot.log_path;
          if (typeof logPath === "string") {
            next.logPath = logPath;
          }
          return next;
        });

        if (!snapshot.running || !snapshot.exists) {
          void refreshRuns();
        }
        const statusValue = typeof snapshot.status === "string" ? snapshot.status : undefined;
        if ((!snapshot.running || !snapshot.exists) && statusValue === "error") {
          setPendingRunSelectionBaseline(undefined);
        }
      } catch (error) {
        setJob((current) => {
          if (!current || current.jobId !== activeJobId) {
            return current;
          }
          return {
            ...current,
            status: "error",
            lastMessage: formatError(error),
          };
        });
      }
    },
    [job, refreshRuns],
  );

  const handleStartPipeline = useCallback(
    async (payload: PipelineRunRequest) => {
      const baseline = lastRunPath ?? (runs[0]?.path ?? null);
      setPendingRunSelectionBaseline(baseline);
      try {
        const response = await startPipelineRun(payload);
        setJob({
          jobId: response.job_id,
          status: "running",
          lastMessage: "Pipeline run started.",
          lastUpdated: Date.now(),
          log: "",
          sharpeBest: null,
          progress: null,
          summaries: [],
        });
        setBanner("Pipeline run launched.");
        void refreshJob(response.job_id);
      } catch (error) {
        setPendingRunSelectionBaseline(undefined);
        throw error;
      }
    },
    [lastRunPath, runs, refreshJob],
  );

  usePolling(() => {
    void refreshJob();
  }, 2000, Boolean(job && job.status === "running"));

  usePolling(() => {
    if (!runsLoading) {
      void refreshRuns();
    }
  }, 1500, pendingRunSelectionBaseline !== undefined);

  const handleRelabel = useCallback(
    async (run: RunSummary, label: string) => {
      try {
        await updateRunLabel({ path: run.path, label });
        await refreshRuns();
        setBanner(label ? `Label updated for ${run.name}` : `Label cleared for ${run.name}`);
      } catch (error) {
        setBanner(formatError(error));
      }
    },
    [refreshRuns],
  );

  const handleCopyLog = useCallback((state: PipelineJobState) => {
    if (!navigator.clipboard) {
      setBanner("Clipboard API unavailable in this browser.");
      return;
    }
    void navigator.clipboard
      .writeText(state.log)
      .then(() => setBanner("Copied pipeline log to clipboard."))
      .catch((error) => setBanner(formatError(error)));
  }, []);

  const handleStopJob = useCallback(async (state: PipelineJobState) => {
    try {
      await stopPipelineJob(state.jobId);
      setJob((current) => {
        if (!current || current.jobId !== state.jobId) {
          return current;
        }
        return {
          ...current,
          lastMessage: "Stop requested…",
        };
      });
      setBanner("Stop signal sent to pipeline run.");
    } catch (error) {
      setBanner(formatError(error));
    }
  }, []);

  return (
    <div className="app-shell" data-test="app-shell">
      <header className="app-shell__header">
        <h1>Alpha Evolve Dashboard</h1>
        <p className="app-shell__subtitle">by LIKIO</p>
        <HeaderNav active={activeTab} onChange={setActiveTab} />
        {banner ? (
          <div className="app-banner" role="status">
            {banner}
            <button className="btn btn--link" onClick={() => setBanner(null)}>
              Dismiss
            </button>
          </div>
        ) : null}
      </header>

      <main className="app-shell__main" data-test="app-main">
        {activeTab === "overview" ? (
          <>
            <div className="runner-shell" data-test="overview-runner">
              <RunnerCanvas />
            </div>
            <div className="app-layout" data-test="overview-layout">
              <div className="app-sidebar">
                <PipelineControls onSubmit={handleStartPipeline} busy={Boolean(job && job.status === "running")} />
                <RunList
                  runs={runs}
                  selected={selectedRunPath}
                  onSelect={handleSelectRun}
                  onRelabel={handleRelabel}
                  onRefresh={() => void refreshRuns()}
                  loading={runsLoading}
                />
                {runsError ? <p className="muted error-text">{runsError}</p> : null}
              </div>

              <div className="app-content">
                {selectedRun ? (
                  <div className="selected-run-meta">
                    <h2>{selectedRun.label || selectedRun.name}</h2>
                  </div>
                ) : (
                  <h2>Select a run to inspect its backtest results</h2>
                )}

                {backtestError ? <p className="muted error-text">{backtestError}</p> : null}
                {backtestLoading ? <p className="muted">Fetching backtest summary…</p> : null}

                <BacktestTable
                  rows={backtestRows}
                  selected={selectedAlphaKey}
                  onSelect={handleSelectRow}
                />

                {timeseriesError ? <p className="muted error-text">{timeseriesError}</p> : null}
                {timeseriesLoading ? <p className="muted">Loading timeseries…</p> : null}

                <TimeseriesCharts
                  data={alphaTimeseries}
                  label={selectedRow?.AlphaID ?? selectedRow?.TimeseriesFile ?? null}
                />

                <JobConsole job={job} onCopyLog={handleCopyLog} onStop={handleStopJob} />
              </div>
            </div>
          </>
        ) : (
          <div className="app-layout app-layout--full" data-test="settings-layout">
            <SettingsPanel onNotify={(msg) => setBanner(msg)} />
          </div>
        )}
      </main>
    </div>
  );
}
