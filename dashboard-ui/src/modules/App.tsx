import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchAlphaTimeseries,
  fetchBacktestSummary,
  fetchJobLog,
  fetchJobStatus,
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
  PipelineJobState,
  PipelineRunRequest,
  RunSummary,
} from "./types";

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function extractSharpeFromLog(log: string): number | null {
  const matches = [...log.matchAll(/Sharpe\(best\)\s*=\s*([+\-]?[0-9.]+)/g)];
  if (!matches.length) {
    return null;
  }
  const last = matches[matches.length - 1];
  const value = Number(last[1]);
  return Number.isFinite(value) ? value : null;
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

  const selectedRun = useMemo(() => {
    if (!selectedRunPath) {
      return null;
    }
    return runs.find((run) => run.path === selectedRunPath) ?? null;
  }, [runs, selectedRunPath]);

  const selectedAlphaKey = useMemo(() => {
    if (!selectedRow) return null;
    return selectedRow.TimeseriesFile || selectedRow.AlphaID || null;
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
      return;
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

    const fallback = runs[0]?.path ?? null;
    if (fallback !== selectedRunPath) {
      setSelectedRunPath(fallback);
    }
  }, [runs, selectedRunPath, lastRunPath]);

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

  const handleStartPipeline = useCallback(
    async (payload: PipelineRunRequest) => {
      const response = await startPipelineRun(payload);
      setJob({
        jobId: response.job_id,
        status: "running",
        lastMessage: "Pipeline run started.",
        lastUpdated: Date.now(),
        log: "",
      });
      setBanner("Pipeline run launched.");
    },
    [],
  );

  const refreshJob = useCallback(async () => {
    setJob((current) => {
      if (!current) return current;
      return { ...current, lastUpdated: Date.now() };
    });
    if (!job) return;

    try {
      const [status, logResponse] = await Promise.all([fetchJobStatus(job.jobId), fetchJobLog(job.jobId)]);

      setJob((current) => {
        if (!current || current.jobId !== job.jobId) return current;
        const next: PipelineJobState = {
          ...current,
          log: logResponse.log ?? "",
          lastUpdated: Date.now(),
        };

        const sharpe = extractSharpeFromLog(logResponse.log ?? "");
        if (sharpe !== null) {
          next.sharpeBest = sharpe;
        }

        if (!status.exists) {
          next.status = "error";
          next.lastMessage = "Job no longer exists.";
        } else if (status.running) {
          next.status = "running";
          next.lastMessage = "Pipeline running…";
        } else {
          next.status = "complete";
          next.lastMessage = "Pipeline finished.";
        }
        return next;
      });

      if (!status.running) {
        void refreshRuns();
      }
    } catch (error) {
      setJob((current) => {
        if (!current) return current;
        return {
          ...current,
          status: "error",
          lastMessage: formatError(error),
        };
      });
    }
  }, [job, refreshRuns]);

  usePolling(() => {
    void refreshJob();
  }, 2000, Boolean(job && job.status === "running"));

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
    <div className="app-shell">
      <header className="app-shell__header">
        <h1>Alpha Evolve Dashboard</h1>
        <p className="app-shell__subtitle">
          React + TypeScript dashboard for evolving and monitoring alpha runs.
        </p>
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

      <main className="app-shell__main">
        {activeTab === "overview" ? (
          <>
            <div className="runner-shell">
              <RunnerCanvas />
            </div>
            <div className="app-layout">
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
                    <span className="muted">{selectedRun.path}</span>
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
          <div className="app-layout">
            <SettingsPanel onNotify={(msg) => setBanner(msg)} />
          </div>
        )}
      </main>
    </div>
  );
}
