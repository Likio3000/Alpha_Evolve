import React, { useCallback, useMemo, useState } from "react";
import {
  PipelineJobState,
  RunSummary,
  BacktestRow,
} from "./types"; // Ensure TabId is exported from HeaderNav or types
import { PipelineControls } from "./components/PipelineControls";
import { RunList } from "./components/RunList"; // This is now the new Shadcn version
import { BacktestTable } from "./components/BacktestTable";
import { TimeseriesCharts } from "./components/TimeseriesCharts";
import { JobConsole } from "./components/JobConsole";
import { RunForensicsPanel } from "./components/RunForensicsPanel";
import { HeaderNav, TabId } from "./components/HeaderNav";
import { RunnerCanvas } from "./components/RunnerCanvas";
import { SettingsPanel } from "./components/SettingsPanel";
import { IntroductionPage } from "./components/IntroductionPage";
import { Zap } from "lucide-react";

// Hooks
import {
  useRuns,
  useBacktestSummary,
  useAlphaTimeseries,
  useStartPipeline,
  useStopJob,
  useUpdateRunLabel
} from "@/hooks/use-dashboard";
import { usePipelineStream } from "@/hooks/use-pipeline-stream";

// Shadcn UI (Optional: Use directly or through composed/refactored components)
// We will use standard divs with Tailwind classes for layout

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export function App(): React.ReactElement {
  const [activeTab, setActiveTab] = useState<TabId>("introduction");
  const [selectedRunPath, setSelectedRunPath] = useState<string | null>(null);
  const [selectedRow, setSelectedRow] = useState<BacktestRow | null>(null);
  const [banner, setBanner] = useState<string | null>(null);

  // Queries
  const { data: runs = [], isLoading: runsLoading, error: runsError } = useRuns();
  const { data: backtestRows = [], isLoading: backtestLoading, error: backtestError } = useBacktestSummary(selectedRunPath);
  const selectedAlphaKey = useMemo(() => {
    if (!selectedRow) return null;
    return selectedRow.TimeseriesFile || selectedRow.TS || selectedRow.AlphaID || null;
  }, [selectedRow]);

  const { data: alphaTimeseries, isLoading: timeseriesLoading, error: timeseriesError } = useAlphaTimeseries(
    selectedRunPath,
    selectedRow?.AlphaID ?? undefined,
    selectedRow?.TimeseriesFile ?? undefined
  );

  // Mutations
  const startPipeline = useStartPipeline();
  const stopJob = useStopJob();
  const updateLabel = useUpdateRunLabel();

  // Pipeline Logic
  // We need to track the "active" job ID.
  // Ideally this comes from the startPipeline response or global state.
  // For now, we mimic the old behavior where startPipeline sets the active ID.
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const { job, streamState, setJob } = usePipelineStream(activeJobId);

  // Computed
  const selectedRun = useMemo(() => {
    if (!selectedRunPath) return null;
    return runs.find((run) => run.path === selectedRunPath) ?? null;
  }, [runs, selectedRunPath]);

  // Handlers
  const handleSelectRun = useCallback((run: RunSummary) => {
    setSelectedRunPath(run.path);
    setSelectedRow(null); // Reset row selection
  }, []);

  const handleSelectRow = useCallback((row: BacktestRow) => {
    setSelectedRow(row);
  }, []);

  const handleStartPipeline = useCallback(async (payload: any) => {
    try {
      const response = await startPipeline.mutateAsync(payload);
      setActiveJobId(response.job_id);
      // Initialize job state immediately for better UX
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
      setActiveTab("controls"); // Switch to controls tab
    } catch (error) {
      setBanner(formatError(error));
    }
  }, [startPipeline, setJob]);

  const handleStopJob = useCallback(async (state: PipelineJobState) => {
    try {
      await stopJob.mutateAsync(state.jobId);
      setBanner("Stop signal sent.");
    } catch (error) {
      setBanner(formatError(error));
    }
  }, [stopJob]);

  const handleRelabel = useCallback((run: RunSummary, label: string) => {
    updateLabel.mutate({ path: run.path, label }, {
      onSuccess: () => setBanner(`Label updated for ${run.name}`),
      onError: (err) => setBanner(formatError(err))
    });
  }, [updateLabel]);

  const handleCopyLog = useCallback((state: PipelineJobState) => {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(state.log)
      .then(() => setBanner("Log copied."))
      .catch((e) => setBanner(formatError(e)));
  }, []);

  return (
    <div className="min-h-screen font-sans text-foreground flex flex-col antialiased relative overflow-x-hidden">
      {/* Background decoration */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[10%] right-[0%] w-[30%] h-[30%] bg-blue-600/10 blur-[100px] rounded-full" />
      </div>

      {/* Header - Floating Glass Design */}
      <header className="sticky top-0 z-50 w-full mb-2">
        <div className="absolute inset-0 bg-background/40 backdrop-blur-xl border-b border-white/5" />
        <div className="container relative flex h-16 items-center px-6 mx-auto max-w-screen-2xl">
          <div className="mr-12 hidden md:flex items-center gap-3 group cursor-default">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-primary to-blue-400 p-[1px] shadow-lg shadow-primary/20 group-hover:rotate-6 transition-transform duration-500">
              <div className="w-full h-full rounded-[11px] bg-background flex items-center justify-center">
                <Zap className="w-5 h-5 text-primary fill-primary/10" />
              </div>
            </div>
            <div className="flex flex-col">
              <h1 className="text-lg font-heading font-bold tracking-tight leading-none">
                Alpha Evolve
              </h1>
              <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground font-bold mt-1">Research Hub</span>
            </div>
          </div>
          <div className="flex-1 flex justify-center">
            <HeaderNav active={activeTab} onChange={setActiveTab as any} />
          </div>
          <div className="ml-auto flex items-center gap-4">
            <div className="hidden sm:flex flex-col items-end mr-2">
              <span className="text-[10px] text-muted-foreground font-bold uppercase tracking-wider">System Status</span>
              <span className="text-[11px] text-green-400 flex items-center gap-1.5 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Operational
              </span>
            </div>
            <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-primary to-purple-500 shadow-lg shadow-primary/20 border border-white/10 p-[1px]">
              <div className="w-full h-full rounded-full bg-background flex items-center justify-center text-[10px] font-bold">AB</div>
            </div>
          </div>
        </div>
        {banner && (
          <div className="bg-primary/5 border-b border-primary/20 p-2 text-center text-sm flex justify-center items-center gap-4 animate-in slide-in-from-top duration-300">
            <span className="text-primary font-bold text-xs uppercase tracking-wider">{banner}</span>
            <button className="text-muted-foreground hover:text-foreground transition-colors font-bold text-[10px] uppercase tracking-wide border border-white/10 px-2 py-0.5 rounded cursor-pointer" onClick={() => setBanner(null)}>Dismiss</button>
          </div>
        )}
      </header>

      <main className="flex-1 container pb-20 max-w-screen-2xl mx-auto px-6 gap-8 flex flex-col pt-8">
        {activeTab === "introduction" && (
          <div className="space-y-6 animate-in fade-in-50 duration-500">
            <div className="h-[200px] w-full rounded-xl border bg-card text-card-foreground shadow overflow-hidden relative">
              <RunnerCanvas />
              <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent pointer-events-none" />
            </div>
            <IntroductionPage />
          </div>
        )}

        {activeTab === "overview" && (
          <div className="grid lg:grid-cols-[380px_1fr] gap-8 animate-in fade-in slide-in-from-bottom-2 duration-700 items-start">
            <div className="lg:sticky lg:top-24 h-[calc(100vh-10rem)] flex flex-col gap-4">
              <div className="flex-1 overflow-hidden glass-panel rounded-2xl border-white/5 shadow-2xl relative">
                <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />
                <RunList
                  runs={runs}
                  selected={selectedRunPath}
                  onSelect={handleSelectRun}
                  onRelabel={handleRelabel}
                  onRefresh={() => { }}
                  loading={runsLoading}
                />
              </div>
              {runsError && (
                <div className="p-3 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-xs font-mono">
                  {formatError(runsError)}
                </div>
              )}
            </div>

            <div className="space-y-8">
              {selectedRun ? (
                <div className="flex flex-col gap-1 p-4 rounded-2xl bg-white/5 border border-white/5">
                  <h2 className="text-3xl font-heading font-bold tracking-tight text-white group flex items-center gap-3">
                    {selectedRun.label || selectedRun.name}
                    <div className="w-2 h-2 rounded-full bg-primary shadow-[0_0_10px_rgba(59,130,246,0.5)]" />
                  </h2>
                  <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">{selectedRun.path}</div>
                </div>
	              ) : (
	                <div className="flex h-[300px] w-full items-center justify-center rounded-[2rem] border border-dashed border-white/10 bg-white/[0.02] group transition-all duration-500 hover:bg-white/[0.04]">
	                  <div className="flex flex-col items-center gap-4 text-center">
	                    <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center border border-white/10 group-hover:rotate-12 transition-transform duration-500">
	                      <Zap className="w-8 h-8 text-muted-foreground" />
	                    </div>
	                    <div className="space-y-1">
	                      <h3 className="text-xl font-bold">No Run Selected</h3>
	                      <p className="text-sm text-muted-foreground max-w-[200px]">Select a run from the sidebar to view detailed analytics.</p>
	                    </div>
	                  </div>
	                </div>
	              )}

              {backtestLoading && (
                <div className="flex items-center gap-3 text-muted-foreground text-sm font-mono animate-pulse">
                  <div className="w-2 h-2 rounded-full bg-primary" />
                  Hydrating backtest data...
                </div>
              )}
              {backtestError && (
                <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-sm">
                  {formatError(backtestError)}
                </div>
              )}

              {selectedRun && (
                <div className="space-y-8 animate-in fade-in duration-500">
                  <div className="glass-panel overflow-hidden rounded-2xl border-white/5 shadow-xl">
                    <BacktestTable
                      rows={backtestRows}
                      selected={selectedAlphaKey}
                      onSelect={handleSelectRow}
                    />
                  </div>

                  <div className="glass-panel p-6 rounded-2xl border-white/5 shadow-xl relative min-h-[450px]">
                    <div className="absolute top-4 right-6 text-[10px] font-mono text-muted-foreground uppercase tracking-widest bg-white/5 px-2 py-1 rounded">Visual Diagnostics</div>
                    {timeseriesLoading ? (
                      <div className="flex flex-col items-center justify-center h-[300px] gap-4">
                        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                        <span className="text-muted-foreground text-xs font-mono uppercase tracking-widest">Rendering Timeseries</span>
                      </div>
                    ) : (
                      <div className="pt-4">
                        <TimeseriesCharts
                          data={alphaTimeseries ?? null}
                          label={selectedRow?.AlphaID ?? selectedRow?.TimeseriesFile ?? null}
                        />
                      </div>
                    )}
                    {timeseriesError && <p className="text-destructive text-sm">{formatError(timeseriesError)}</p>}
                  </div>

                  <div className="glass-panel rounded-2xl border-white/5 shadow-xl overflow-hidden">
                    <RunForensicsPanel runDir={selectedRunPath} />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "controls" && (
          <div className="grid gap-8 animate-in fade-in slide-in-from-bottom-2 duration-700">
            <div className="h-[200px] w-full rounded-2xl border border-white/5 bg-card text-card-foreground shadow-2xl overflow-hidden relative">
              <RunnerCanvas />
              <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent pointer-events-none" />
              <div className="absolute bottom-6 left-8">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                  <h3 className="text-xl font-bold tracking-tight">Active Runner</h3>
                </div>
                <p className="text-sm text-muted-foreground mt-1">Real-time evolutionary synthesis engine diagnostics.</p>
              </div>
            </div>
            <div className="grid lg:grid-cols-[1fr_450px] gap-8 items-start">
              <div className="glass-panel p-6 rounded-2xl border-white/5 shadow-xl">
                <JobConsole job={job} connectionState={streamState} onCopyLog={handleCopyLog} onStop={handleStopJob} />
              </div>
              <div className="glass-panel p-6 rounded-2xl border-white/5 shadow-xl sticky top-24">
                <PipelineControls onSubmit={handleStartPipeline} busy={Boolean(job && job.status === "running")} />
              </div>
            </div>
          </div>
        )}

        {activeTab === "settings" && (
          <div className="animate-in fade-in slide-in-from-bottom-2 duration-700">
            <div className="glass-panel p-8 rounded-[2rem] border-white/5 shadow-2xl">
              <SettingsPanel onNotify={(msg) => setBanner(msg)} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
