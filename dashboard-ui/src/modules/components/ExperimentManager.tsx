import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  decideExperimentProposal,
  exportBestExperimentConfig,
  fetchExperimentIterations,
  fetchExperimentProposals,
  fetchExperimentSearchSpaces,
  fetchExperimentSessions,
  startExperimentSession,
  stopExperimentSession,
} from "../api";
import { ExperimentIteration, ExperimentProposal, ExperimentSession, PipelineRunRequest } from "../types";
import { usePolling } from "../hooks/usePolling";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

interface ExperimentManagerProps {
  onNotify: (message: string) => void;
  onReplayPipeline: (payload: PipelineRunRequest) => Promise<void>;
}

export function ExperimentManager({ onNotify, onReplayPipeline }: ExperimentManagerProps): React.ReactElement {
  const [searchSpaces, setSearchSpaces] = useState<string[]>([]);
  const [sessions, setSessions] = useState<ExperimentSession[]>([]);
  const [sessionsError, setSessionsError] = useState<string | null>(null);

  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [iterations, setIterations] = useState<ExperimentIteration[]>([]);
  const [proposals, setProposals] = useState<ExperimentProposal[]>([]);
  const [detailsError, setDetailsError] = useState<string | null>(null);
  const detailsCacheRef = useRef(
    new Map<string, { iterations: ExperimentIteration[]; proposals: ExperimentProposal[] }>(),
  );
  const detailsRequestIdRef = useRef(0);

  const [startSearchSpace, setStartSearchSpace] = useState<string>("");
  const [startConfig, setStartConfig] = useState<string>("configs/sp500.toml");
  const [startIterations, setStartIterations] = useState<number>(5);
  const [startSeed, setStartSeed] = useState<number>(0);
  const [startAutoApprove, setStartAutoApprove] = useState<boolean>(false);
  const [startBusy, setStartBusy] = useState(false);

  const selectedSession = useMemo(() => {
    if (!selectedSessionId) return null;
    return sessions.find((s) => s.session_id === selectedSessionId) ?? null;
  }, [sessions, selectedSessionId]);

  const pendingProposal = useMemo(() => {
    return proposals.find((p) => p.status === "pending") ?? null;
  }, [proposals]);

  const refreshSearchSpaces = useCallback(async () => {
    try {
      const items = await fetchExperimentSearchSpaces();
      setSearchSpaces(items);
      if (!startSearchSpace && items.length) {
        setStartSearchSpace(items[0]);
      }
    } catch (error) {
      // Non-fatal; keep manual input usable.
      setSearchSpaces([]);
    }
  }, [startSearchSpace]);

  const refreshSessions = useCallback(async () => {
    try {
      const payload = await fetchExperimentSessions(200);
      setSessions(payload.items ?? []);
      setSessionsError(null);
      if (!selectedSessionId && payload.items?.length) {
        setSelectedSessionId(payload.items[0].session_id);
      }
    } catch (error) {
      setSessionsError(formatError(error));
    }
  }, [selectedSessionId]);

  const refreshDetails = useCallback(async () => {
    if (!selectedSessionId) {
      setIterations([]);
      setProposals([]);
      return;
    }
    const requestId = (detailsRequestIdRef.current += 1);
    const sessionId = selectedSessionId;
    try {
      const [iters, props] = await Promise.all([
        fetchExperimentIterations(sessionId),
        fetchExperimentProposals(sessionId),
      ]);
      const nextIterations = iters.items ?? [];
      const nextProposals = props.items ?? [];
      detailsCacheRef.current.set(sessionId, { iterations: nextIterations, proposals: nextProposals });
      if (detailsRequestIdRef.current !== requestId) return;

      setIterations(nextIterations);
      setProposals(nextProposals);
      setDetailsError(null);
    } catch (error) {
      if (detailsRequestIdRef.current !== requestId) return;
      setDetailsError(formatError(error));
    }
  }, [selectedSessionId]);

  useEffect(() => {
    void refreshSearchSpaces();
    void refreshSessions();
  }, [refreshSearchSpaces, refreshSessions]);

  usePolling(() => {
    void refreshSessions();
    void refreshDetails();
  }, 2500, true);

  useEffect(() => {
    if (!selectedSessionId) return;
    const cached = detailsCacheRef.current.get(selectedSessionId);
    if (cached) {
      setIterations(cached.iterations);
      setProposals(cached.proposals);
    } else {
      setIterations([]);
      setProposals([]);
    }
    void refreshDetails();
  }, [refreshDetails, selectedSessionId]);

  const handleStartSession = useCallback(async () => {
    if (!startSearchSpace) {
      onNotify("Select a search space to start a session.");
      return;
    }
    setStartBusy(true);
    try {
      const resp = await startExperimentSession({
        search_space: startSearchSpace,
        config: startConfig.trim() ? startConfig.trim() : null,
        iterations: startIterations,
        seed: startSeed,
        objective: "Sharpe",
        minimize: false,
        auto_approve: startAutoApprove,
      });
      setSelectedSessionId(resp.session_id);
      onNotify(`Experiment session started: ${resp.session_id}`);
      await refreshSessions();
      await refreshDetails();
    } catch (error) {
      onNotify(formatError(error));
    } finally {
      setStartBusy(false);
    }
  }, [
    onNotify,
    refreshDetails,
    refreshSessions,
    startAutoApprove,
    startConfig,
    startIterations,
    startSearchSpace,
    startSeed,
  ]);

  const handleDecide = useCallback(
    async (decision: "approved" | "rejected") => {
      if (!pendingProposal || !selectedSessionId) return;
      try {
        await decideExperimentProposal(selectedSessionId, pendingProposal.id, {
          decision,
          decided_by: "dashboard",
        });
        onNotify(`${decision === "approved" ? "Approved" : "Rejected"} proposal #${pendingProposal.id}.`);
        await refreshDetails();
        await refreshSessions();
      } catch (error) {
        onNotify(formatError(error));
      }
    },
    [onNotify, pendingProposal, refreshDetails, refreshSessions, selectedSessionId],
  );

  const handleStop = useCallback(async () => {
    if (!selectedSessionId) return;
    try {
      await stopExperimentSession(selectedSessionId);
      onNotify("Stop requested for experiment session.");
      await refreshSessions();
    } catch (error) {
      onNotify(formatError(error));
    }
  }, [onNotify, refreshSessions, selectedSessionId]);

  const handleReplayBest = useCallback(async () => {
    if (!selectedSessionId) return;
    try {
      const resp = await exportBestExperimentConfig(selectedSessionId);
      await onReplayPipeline({ generations: resp.generations, config: resp.config_path });
      onNotify(`Replaying best config from session ${selectedSessionId}.`);
    } catch (error) {
      onNotify(formatError(error));
    }
  }, [onNotify, onReplayPipeline, selectedSessionId]);

  return (
    <Card className="w-full border-none shadow-none bg-transparent">
      <div className="mb-6">
        <h2 className="text-2xl font-bold tracking-tight">Experiment Manager</h2>
        <p className="text-sm text-muted-foreground">Tracked self-evolution sessions with approval gates and replay support.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Start Session</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Search Space</Label>
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                  value={startSearchSpace}
                  onChange={(e) => setStartSearchSpace(e.target.value)}
                  disabled={startBusy}
                >
                  {searchSpaces.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                  {!searchSpaces.length ? <option value={startSearchSpace}>Enter manually</option> : null}
                </select>
              </div>

              <div className="space-y-2">
                <Label>Base Config (optional)</Label>
                <Input
                  type="text"
                  value={startConfig}
                  onChange={(e) => setStartConfig(e.target.value)}
                  placeholder="configs/sp500.toml"
                  disabled={startBusy}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Iterations</Label>
                  <Input
                    type="number"
                    min={1}
                    value={startIterations}
                    onChange={(e) => setStartIterations(Number(e.target.value))}
                    disabled={startBusy}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Seed</Label>
                  <Input
                    type="number"
                    min={0}
                    value={startSeed}
                    onChange={(e) => setStartSeed(Number(e.target.value))}
                    disabled={startBusy}
                  />
                </div>
              </div>

              <div className="flex items-center space-x-2 pt-2">
                <input
                  type="checkbox"
                  id="auto-approve"
                  className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600"
                  checked={startAutoApprove}
                  onChange={(e) => setStartAutoApprove(e.target.checked)}
                  disabled={startBusy}
                />
                <Label htmlFor="auto-approve" className="font-normal cursor-pointer">Auto approve proposals</Label>
              </div>

              <Button className="w-full" onClick={handleStartSession} disabled={startBusy}>
                {startBusy ? "Starting…" : "Start Session"}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Sessions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {sessionsError ? <p className="text-xs text-destructive bg-destructive/10 p-2 rounded">{sessionsError}</p> : null}

              <div className="space-y-2">
                <Label>Active Session</Label>
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                  value={selectedSessionId ?? ""}
                  onChange={(e) => setSelectedSessionId(e.target.value)}
                >
                  {sessions.map((s) => (
                    <option key={s.session_id} value={s.session_id}>
                      {s.session_id.slice(0, 8)}… ({s.status})
                    </option>
                  ))}
                  {!sessions.length ? <option value="">No sessions</option> : null}
                </select>
              </div>

              <div className="flex gap-2">
                <Button variant="secondary" className="flex-1" onClick={handleStop} disabled={!selectedSessionId}>
                  Stop
                </Button>
                <Button
                  variant="secondary"
                  className="flex-1"
                  onClick={handleReplayBest}
                  disabled={!selectedSession?.best_iteration_id}
                >
                  Replay Best
                </Button>
              </div>
            </CardContent>

            {selectedSession ? (
              <CardFooter className="bg-muted/30 flex flex-col items-start gap-1 p-4 text-xs">
                <div>
                  <span className="text-muted-foreground">Status:</span> <Badge variant="outline" className="ml-1 uppercase text-[10px]">{selectedSession.status}</Badge>{" "}
                  {selectedSession.running_task ? <span className="text-blue-500 font-medium ml-2">(Runner Active)</span> : null}
                </div>
                <div className="flex gap-4 mt-1">
                  <div><span className="text-muted-foreground">Best Sharpe:</span> <span className="font-mono">{selectedSession.best_sharpe ?? "—"}</span></div>
                  <div><span className="text-muted-foreground">Best Corr:</span> <span className="font-mono">{selectedSession.best_corr ?? "—"}</span></div>
                </div>
                {selectedSession.last_error ? (
                  <div className="text-destructive mt-1 font-medium">Error: {selectedSession.last_error}</div>
                ) : null}
              </CardFooter>
            ) : null}
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Approvals</CardTitle>
            </CardHeader>
            <CardContent>
              {detailsError ? <p className="text-xs text-destructive mb-2">{detailsError}</p> : null}
              {pendingProposal ? (
                <div className="space-y-4">
                  <div className="text-sm">
                    <strong>Pending proposal #{pendingProposal.id}</strong> (next iteration {pendingProposal.next_iteration})
                  </div>
                  <div className="bg-muted/50 p-3 rounded-md border font-mono text-xs overflow-auto max-h-[200px]">
                    <pre>{JSON.stringify(pendingProposal.proposed_updates_json, null, 2)}</pre>
                  </div>
                  <div className="flex gap-2">
                    <Button className="flex-1" onClick={() => void handleDecide("approved")}>
                      Approve
                    </Button>
                    <Button variant="secondary" className="flex-1" onClick={() => void handleDecide("rejected")}>
                      Reject
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground py-4 text-center">No pending proposals.</div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Iterations</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {iterations.length ? (
                <div className="max-h-[300px] overflow-y-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[50px]">#</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Sharpe</TableHead>
                        <TableHead>Avg Corr</TableHead>
                        <TableHead className="text-right">Run</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {iterations.map((it) => (
                        <TableRow key={it.id}>
                          <TableCell className="font-medium">{it.iteration_index}</TableCell>
                          <TableCell>
                            <Badge variant="secondary" className="text-[10px] font-normal">{it.status}</Badge>
                          </TableCell>
                          <TableCell>{it.objective_sharpe ?? "—"}</TableCell>
                          <TableCell>{it.objective_corr ?? "—"}</TableCell>
                          <TableCell className="text-right text-xs text-muted-foreground font-mono">
                            {it.run_dir ? String(it.run_dir).split("/").slice(-1)[0] : "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground p-6 text-center">No iterations recorded yet.</div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </Card>
  );
}
