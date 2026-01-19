import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import {
  useCodexSummary,
  useStartCodexWatcher,
  useStopCodexWatcher,
  useUpdateCodexSettings,
} from "@/hooks/use-dashboard";

type CodexModePageProps = {
  onNotify?: (message: string) => void;
};

function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function formatTimestamp(value: number | null | undefined): string {
  if (!value || !Number.isFinite(value)) return "-";
  return new Date(value * 1000).toLocaleTimeString();
}

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export function CodexModePage({ onNotify }: CodexModePageProps): React.ReactElement {
  const { data, isLoading, error } = useCodexSummary();
  const updateSettings = useUpdateCodexSettings();
  const startWatcher = useStartCodexWatcher();
  const stopWatcher = useStopCodexWatcher();

  const [notify, setNotify] = useState(false);
  const [reviewInterval, setReviewInterval] = useState("5");
  const [sleepSeconds, setSleepSeconds] = useState("15");
  const [yoloMode, setYoloMode] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const [autoRunCommand, setAutoRunCommand] = useState("codex");
  const [autoRunMode, setAutoRunMode] = useState("terminal");
  const [autoRunCooldown, setAutoRunCooldown] = useState("300");
  const [showHelp, setShowHelp] = useState(false);

  const settings = data?.settings;
  const watcher = data?.watcher;
  const state = data?.state;

  useEffect(() => {
    if (!settings) return;
    setNotify(Boolean(settings.notify));
    setReviewInterval(String(settings.review_interval ?? 5));
    setSleepSeconds(String(settings.sleep_seconds ?? 15));
    setYoloMode(Boolean(settings.yolo_mode));
    setAutoRun(Boolean(settings.auto_run));
    setAutoRunCommand(settings.auto_run_command ?? "codex");
    setAutoRunMode(settings.auto_run_mode ?? "terminal");
    setAutoRunCooldown(String(settings.auto_run_cooldown ?? 300));
  }, [settings]);

  const runsSinceReview = state?.runs_since_review ?? 0;
  const lastAutorun = state?.last_autorun_ts ?? null;
  const intervalNumber = Number(reviewInterval);
  const runsToReview = Number.isFinite(intervalNumber) && intervalNumber > 0
    ? Math.max(0, intervalNumber - runsSinceReview)
    : null;

  const events = useMemo(() => {
    const items = data?.events ?? [];
    return [...items].reverse();
  }, [data?.events]);

  const handleSave = useCallback(async () => {
    const interval = Number(reviewInterval);
    const sleep = Number(sleepSeconds);
    if (!Number.isFinite(interval) || interval <= 0) {
      onNotify?.("Review interval must be a positive number.");
      return;
    }
    if (!Number.isFinite(sleep) || sleep <= 0) {
      onNotify?.("Sleep seconds must be a positive number.");
      return;
    }
    try {
      await updateSettings.mutateAsync({
        notify,
        review_interval: interval,
        sleep_seconds: sleep,
        yolo_mode: yoloMode,
        auto_run: autoRun,
        auto_run_command: autoRunCommand,
        auto_run_mode: autoRunMode,
        auto_run_cooldown: Number(autoRunCooldown) || 300,
      });
      onNotify?.("Codex settings updated.");
    } catch (err) {
      onNotify?.(formatError(err));
    }
  }, [
    autoRun,
    autoRunCommand,
    autoRunCooldown,
    autoRunMode,
    notify,
    onNotify,
    reviewInterval,
    sleepSeconds,
    updateSettings,
    yoloMode,
  ]);

  const handleStart = useCallback(async () => {
    try {
      const resp = await startWatcher.mutateAsync();
      if (resp.started) {
        onNotify?.("Codex watcher started.");
      } else {
        onNotify?.(resp.detail || "Codex watcher start failed.");
      }
    } catch (err) {
      onNotify?.(formatError(err));
    }
  }, [onNotify, startWatcher]);

  const handleStop = useCallback(async () => {
    try {
      const resp = await stopWatcher.mutateAsync();
      if (resp.stopped) {
        onNotify?.("Codex watcher stopped.");
      } else {
        onNotify?.(resp.detail || "Codex watcher stop failed.");
      }
    } catch (err) {
      onNotify?.(formatError(err));
    }
  }, [onNotify, stopWatcher]);

  const reviewNeeded = Boolean(data?.review_needed);

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-2 duration-700">
      <div className="grid lg:grid-cols-[1.1fr_1fr] gap-8">
        <Card className="glass-panel border-white/10 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-transparent pointer-events-none" />
          <CardHeader className="relative flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-heading font-bold tracking-tight">Codex Mode</CardTitle>
              <p className="text-sm text-muted-foreground">Live loop for Sharpe optimization.</p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => setShowHelp((prev) => !prev)}>
                {showHelp ? "Hide Help" : "How it works"}
              </Button>
              <Badge variant={watcher?.running ? "default" : "secondary"} className="text-[10px] uppercase tracking-widest">
                {watcher?.running ? "running" : "idle"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="relative space-y-6">
            {showHelp && (
              <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 text-xs text-muted-foreground space-y-2">
                <div className="text-sm font-semibold text-foreground">How Codex Mode works</div>
                <div>- The watcher polls run folders for completed pipeline and ML runs.</div>
                <div>- It only emits events after a run finishes (pipeline or ML).</div>
                <div>- Each event updates the inbox and experiment log, plus a desktop notification.</div>
                <div>- Every N runs it raises a review reminder to check uncommitted changes.</div>
                <div>- Auto-run launches Codex when a run completes (cooldown prevents spam).</div>
              </div>
            )}
            {reviewNeeded && (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4">
                <div className="text-sm font-semibold text-destructive">Review Needed</div>
                <p className="text-xs text-destructive mt-1">
                  Review cadence reached. Inspect uncommitted changes before the next run.
                </p>
              </div>
            )}

            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-widest text-muted-foreground">Watcher</span>
                  <Badge variant="outline" className="text-[10px] uppercase tracking-widest">
                    PID {watcher?.pid ?? "-"}
                  </Badge>
                </div>
                <div className="text-sm font-semibold">
                  {watcher?.running ? "Active" : "Stopped"}
                </div>
                <div className="text-xs text-muted-foreground">
                  Last update: {data?.updated_at ?? "-"}
                </div>
                <div className="text-xs text-muted-foreground">
                  Last scan: {formatTimestamp(state?.last_scan ?? null)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Last auto-run: {formatTimestamp(lastAutorun)}
                </div>
              </div>
              <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-widest text-muted-foreground">Review Cadence</span>
                  <Badge variant="outline" className="text-[10px] uppercase tracking-widest">
                    {runsSinceReview}/{intervalNumber || "-"}
                  </Badge>
                </div>
                <div className="text-sm font-semibold">
                  {runsToReview === null ? "Disabled" : `${runsToReview} runs to review`}
                </div>
                <div className="text-xs text-muted-foreground">
                  Runs seen: {(state?.seen_pipeline_runs?.length ?? 0) + (state?.seen_ml_runs?.length ?? 0)}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Button onClick={handleStart} disabled={Boolean(watcher?.running)}>
                Start Watcher
              </Button>
              <Button variant="ghost" onClick={handleStop} disabled={!watcher?.running}>
                Stop Watcher
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-panel border-white/10">
          <CardHeader>
            <CardTitle className="text-xl font-bold">Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3">
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.04] p-3">
                <div>
                  <div className="text-sm font-semibold">Notifications</div>
                  <div className="text-xs text-muted-foreground">macOS notifications on run completion.</div>
                </div>
                <Button
                  size="sm"
                  variant={notify ? "default" : "secondary"}
                  onClick={() => setNotify((prev) => !prev)}
                >
                  {notify ? "On" : "Off"}
                </Button>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.04] p-3">
                <div>
                  <div className="text-sm font-semibold">YOLO Mode</div>
                  <div className="text-xs text-muted-foreground">Allow aggressive code/UI/test edits.</div>
                </div>
                <Button
                  size="sm"
                  variant={yoloMode ? "default" : "secondary"}
                  onClick={() => setYoloMode((prev) => !prev)}
                >
                  {yoloMode ? "Enabled" : "Off"}
                </Button>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.04] p-3">
                <div>
                  <div className="text-sm font-semibold">Auto-run Codex</div>
                  <div className="text-xs text-muted-foreground">Launch Codex automatically after runs finish.</div>
                </div>
                <Button
                  size="sm"
                  variant={autoRun ? "default" : "secondary"}
                  onClick={() => setAutoRun((prev) => !prev)}
                >
                  {autoRun ? "Enabled" : "Off"}
                </Button>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="grid gap-2">
                <Label>Review interval (runs)</Label>
                <Input value={reviewInterval} onChange={(e) => setReviewInterval(e.target.value)} />
              </div>
              <div className="grid gap-2">
                <Label>Sleep seconds</Label>
                <Input value={sleepSeconds} onChange={(e) => setSleepSeconds(e.target.value)} />
              </div>
            </div>

            <div className="grid gap-3">
              <div className="grid gap-2">
                <Label>Auto-run command</Label>
                <Input
                  value={autoRunCommand}
                  onChange={(e) => setAutoRunCommand(e.target.value)}
                  placeholder="codex --prompt-file {prompt_file}"
                />
                <div className="text-[11px] text-muted-foreground">
                  Use <span className="font-mono">{`{prompt_file}`}</span> to inject the session prompt.
                </div>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <div className="grid gap-2">
                  <Label>Auto-run mode</Label>
                  <Input value={autoRunMode} onChange={(e) => setAutoRunMode(e.target.value)} placeholder="terminal" />
                </div>
                <div className="grid gap-2">
                  <Label>Auto-run cooldown (sec)</Label>
                  <Input value={autoRunCooldown} onChange={(e) => setAutoRunCooldown(e.target.value)} />
                </div>
              </div>
            </div>

            <Button onClick={handleSave} disabled={updateSettings.isPending}>
              {updateSettings.isPending ? "Saving..." : "Save Settings"}
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="grid lg:grid-cols-[1.2fr_1fr] gap-8">
        <Card className="glass-panel border-white/10">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Recent Events</CardTitle>
          </CardHeader>
          <CardContent>
            {error && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
                {formatError(error)}
              </div>
            )}
            {isLoading && <div className="text-xs text-muted-foreground">Loading events...</div>}
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Kind</TableHead>
                  <TableHead>Run</TableHead>
                  <TableHead>Sharpe</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {events.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} className="text-xs text-muted-foreground">
                      No events yet.
                    </TableCell>
                  </TableRow>
                )}
                {events.map((event, index) => (
                  <TableRow key={`${event.run_name ?? event.label ?? "event"}-${index}`}>
                    <TableCell className="text-xs text-muted-foreground">{event.ts ?? "-"}</TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="text-[10px] uppercase tracking-widest">
                        {event.kind ?? "-"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs">
                      <div className="font-semibold">{event.run_name ?? event.label ?? "-"}</div>
                      <div className="text-[10px] text-muted-foreground truncate">{event.run_dir ?? ""}</div>
                    </TableCell>
                    <TableCell className="text-xs font-mono">{formatNumber(event.sharpe, 3)}</TableCell>
                    <TableCell className="text-xs">{event.status ?? "-"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card className="glass-panel border-white/10">
            <CardHeader>
              <CardTitle className="text-lg font-semibold">Codex Inbox</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className={cn("whitespace-pre-wrap text-xs text-muted-foreground", !data?.inbox && "opacity-60")}>
                {data?.inbox ?? "No inbox message yet."}
              </pre>
            </CardContent>
          </Card>

          <Card className="glass-panel border-white/10">
            <CardHeader>
              <CardTitle className="text-lg font-semibold">Session Prompt</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className={cn("whitespace-pre-wrap text-xs text-muted-foreground", !data?.session_prompt && "opacity-60")}>
                {data?.session_prompt ?? "Session prompt not available."}
              </pre>
              <div className="mt-3 text-[11px] text-muted-foreground">
                Source: <span className="font-mono">docs/codex_mode/SESSION_PROMPT.md</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Card className="glass-panel border-white/10">
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Experiment Log (tail)</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className={cn("whitespace-pre-wrap text-xs text-muted-foreground", !data?.experiments && "opacity-60")}>
            {data?.experiments ?? "No experiment log yet."}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
