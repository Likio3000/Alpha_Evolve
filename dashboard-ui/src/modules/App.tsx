import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { RunForensicsPanel } from "./components/RunForensicsPanel";
import { HeaderNav, TabId } from "./components/HeaderNav";
import { RunnerCanvas } from "./components/RunnerCanvas";
import { SettingsPanel } from "./components/SettingsPanel";
import { IntroductionPage } from "./components/IntroductionPage";
import { ExperimentManager } from "./components/ExperimentManager";
import { usePolling } from "./hooks/usePolling";
import {
  AlphaTimeseries,
  BacktestRow,
  GenerationSummary,
  PipelineJobState,
  PipelineRunRequest,
  RunSummary,
} from "./types";
import { mapGenerationProgress, mapGenerationSummary } from "./pipelineMapping";

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

const SUMMARY_HISTORY_LIMIT = 400;
const LOG_HISTORY_LIMIT_CHARS = 50_000;

type StreamState = "connected" | "retrying" | "stale";

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
  const [streamState, setStreamState] = useState<StreamState | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const lastStreamEventAtRef = useRef<number>(0);
  const [banner, setBanner] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("introduction");
  const closeEventStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setStreamState(null);
  }, []);
  const [pendingRunSelectionBaseline, setPendingRunSelectionBaseline] = useState<string | null | undefined>(undefined);

  const activeJobId = job?.jobId ?? null;
  const jobRunning = Boolean(job && job.status === "running");

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
      const jobId = jobIdOverride ?? activeJobId;
      if (!jobId) {
        return;
      }
      try {
        const snapshot = await fetchJobActivity(jobId);
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
            current && current.jobId === jobId
              ? current
              : {
                  jobId,
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
            next.log = snapshot.log;
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
          if (!current || current.jobId !== jobId) {
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
    [activeJobId, refreshRuns],
  );

  const applyPipelineEvent = useCallback(
    (activeJobId: string, raw: string) => {
      let payload: unknown;
      try {
        payload = JSON.parse(raw);
      } catch {
        return;
      }
      if (!payload || typeof payload !== "object") {
        return;
      }
      const event = payload as Record<string, unknown>;
      const eventType = event.type;

      if (eventType === "final" || (eventType === "status" && event.msg === "exit")) {
        void refreshRuns();
      }
      if (eventType === "status" && event.msg === "exit" && Number(event.code) !== 0) {
        setPendingRunSelectionBaseline(undefined);
      }

      setJob((current) => {
        if (!current || current.jobId !== activeJobId) {
          return current;
        }
        const next: PipelineJobState = { ...current, lastUpdated: Date.now() };

        if (eventType === "log") {
          const rawLine = typeof event.raw === "string" ? event.raw : "";
          if (rawLine) {
            const normalized = rawLine.endsWith("\n") ? rawLine : `${rawLine}\n`;
            const combined = `${next.log || ""}${normalized}`;
            next.log =
              combined.length > LOG_HISTORY_LIMIT_CHARS
                ? combined.slice(combined.length - LOG_HISTORY_LIMIT_CHARS)
                : combined;
          }
          return next;
        }

        if (eventType === "progress") {
          const progress = mapGenerationProgress(event.data);
          if (progress) {
            next.progress = progress;
          }
          return next;
        }

        if (eventType === "gen_summary") {
          const summary = mapGenerationSummary(event.data);
          if (summary) {
            const summaries = next.summaries ?? [];
            const idx = summaries.findIndex((entry) => entry.generation === summary.generation);
            const updated = idx >= 0 ? summaries.map((entry, i) => (i === idx ? summary : entry)) : [...summaries, summary];
            updated.sort((a, b) => a.generation - b.generation);
            next.summaries =
              updated.length > SUMMARY_HISTORY_LIMIT ? updated.slice(updated.length - SUMMARY_HISTORY_LIMIT) : updated;
          }
          return next;
        }

        if (eventType === "score") {
          const sharpeRaw = event.sharpe_best ?? event.sharpeBest;
          const sharpe = Number(sharpeRaw);
          if (Number.isFinite(sharpe)) {
            next.sharpeBest = sharpe;
          }
          return next;
        }

        if (eventType === "final") {
          const sharpe = Number(event.sharpe_best ?? event.sharpeBest);
          if (Number.isFinite(sharpe)) {
            next.sharpeBest = sharpe;
          }
          next.status = "complete";
          next.lastMessage = "Pipeline finished.";
          return next;
        }

        if (eventType === "status") {
          const msg = event.msg;
          if (msg === "stop_requested") {
            next.lastMessage = "Stop requested…";
            return next;
          }
          if (msg === "exit") {
            const code = Number(event.code);
            next.status = code === 0 ? "complete" : "error";
            next.lastMessage = code === 0 ? "Pipeline finished." : "Pipeline stopped.";
            return next;
          }
          if (typeof msg === "string" && msg.trim()) {
            next.lastMessage = msg;
          }
          return next;
        }

        if (eventType === "error") {
          next.status = "error";
          const detail = typeof event.detail === "string" ? event.detail : null;
          next.lastMessage = detail && detail.trim() ? detail : "Pipeline error.";
          return next;
        }

        return next;
      });
    },
    [refreshRuns],
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

  useEffect(() => {
    if (!activeJobId || !jobRunning) {
      closeEventStream();
      return;
    }
    if (typeof EventSource === "undefined") {
      setStreamState("stale");
      return;
    }

    const jobId = activeJobId;
    closeEventStream();
    setStreamState("retrying");
    lastStreamEventAtRef.current = Date.now();

    const url = `/api/pipeline/events/${encodeURIComponent(jobId)}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    const touch = () => {
      lastStreamEventAtRef.current = Date.now();
    };

    const markConnected = () => {
      touch();
      setStreamState("connected");
    };

    es.onopen = () => {
      markConnected();
    };

    es.onmessage = (event) => {
      touch();
      applyPipelineEvent(jobId, event.data);
    };

    es.onerror = () => {
      touch();
      setStreamState((prev) => (prev === "connected" ? "retrying" : prev ?? "retrying"));
      void refreshJob(jobId);
    };

    es.addEventListener("ping", markConnected as EventListener);

    const staleInterval = window.setInterval(() => {
      const ageMs = Date.now() - lastStreamEventAtRef.current;
      setStreamState((prev) => {
        if (ageMs > 15_000) {
          return prev === "connected" ? "stale" : prev;
        }
        return prev === "stale" ? "connected" : prev;
      });
    }, 2_000);

    return () => {
      window.clearInterval(staleInterval);
      es.close();
      if (eventSourceRef.current === es) {
        eventSourceRef.current = null;
      }
      setStreamState(null);
    };
  }, [activeJobId, applyPipelineEvent, closeEventStream, jobRunning, refreshJob]);

  usePolling(() => {
    void refreshJob();
  }, 2500, Boolean(job && job.status === "running" && streamState !== "connected"));

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
        <div className="app-shell__header-inner">
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
        </div>
      </header>

      <main className="app-shell__main" data-test="app-main">
        {activeTab === "introduction" ? (
          <>
            <div className="runner-shell" data-test="introduction-runner">
              <RunnerCanvas />
            </div>
            <IntroductionPage />
          </>
        ) : null}

        {activeTab === "overview" ? (
          <>
            <div className="runner-shell" data-test="overview-runner">
              <RunnerCanvas />
            </div>
            <div className="app-layout" data-test="overview-layout">
              <div className="app-sidebar">
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

                <RunForensicsPanel runDir={selectedRunPath} />
              </div>
            </div>
          </>
        ) : null}

        {activeTab === "controls" ? (
          <>
            <div className="runner-shell" data-test="controls-runner">
              <RunnerCanvas />
            </div>
            <div className="app-layout app-layout--stack" data-test="controls-layout">
              <PipelineControls onSubmit={handleStartPipeline} busy={Boolean(job && job.status === "running")} />
              <JobConsole job={job} connectionState={streamState} onCopyLog={handleCopyLog} onStop={handleStopJob} />
            </div>
          </>
        ) : null}

        {activeTab === "settings" ? (
          <div className="app-layout app-layout--full" data-test="settings-layout">
            <SettingsPanel onNotify={(msg) => setBanner(msg)} />
          </div>
        ) : null}

        {activeTab === "experiments" ? (
          <div className="app-layout app-layout--full" data-test="experiments-layout">
            <ExperimentManager
              onNotify={(msg) => setBanner(msg)}
              onReplayPipeline={handleStartPipeline}
            />
          </div>
        ) : null}
      </main>
    </div>
  );
}
