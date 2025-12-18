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
    <section className="panel experiment-panel" data-test="experiment-manager">
      <div className="panel-header">
        <div>
          <h2>Experiment Manager</h2>
          <p className="muted">Tracked self-evolution sessions with approval gates and replay support.</p>
        </div>
      </div>

      <div className="experiment-grid">
        <div className="experiment-col">
          <div className="experiment-section">
            <h3>Start Session</h3>
            <div className="experiment-form">
              <label className="experiment-field">
                <span>Search Space</span>
                <select
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
              </label>

              <label className="experiment-field">
                <span>Base Config (optional)</span>
                <input
                  type="text"
                  value={startConfig}
                  onChange={(e) => setStartConfig(e.target.value)}
                  placeholder="configs/sp500.toml"
                  disabled={startBusy}
                />
              </label>

              <label className="experiment-field">
                <span>Iterations</span>
                <input
                  type="number"
                  min={1}
                  value={startIterations}
                  onChange={(e) => setStartIterations(Number(e.target.value))}
                  disabled={startBusy}
                />
              </label>

              <label className="experiment-field">
                <span>Seed</span>
                <input
                  type="number"
                  min={0}
                  value={startSeed}
                  onChange={(e) => setStartSeed(Number(e.target.value))}
                  disabled={startBusy}
                />
              </label>

              <label className="experiment-checkbox">
                <input
                  type="checkbox"
                  checked={startAutoApprove}
                  onChange={(e) => setStartAutoApprove(e.target.checked)}
                  disabled={startBusy}
                />
                <span>Auto approve proposals</span>
              </label>

              <button className="btn" type="button" onClick={handleStartSession} disabled={startBusy}>
                {startBusy ? "Starting…" : "Start Session"}
              </button>
            </div>
          </div>

          <div className="experiment-section">
            <h3>Sessions</h3>
            {sessionsError ? <p className="muted error-text">{sessionsError}</p> : null}
            <div className="experiment-form">
              <label className="experiment-field">
                <span>Active Session</span>
                <select value={selectedSessionId ?? ""} onChange={(e) => setSelectedSessionId(e.target.value)}>
                  {sessions.map((s) => (
                    <option key={s.session_id} value={s.session_id}>
                      {s.session_id.slice(0, 8)}… ({s.status})
                    </option>
                  ))}
                  {!sessions.length ? <option value="">No sessions</option> : null}
                </select>
              </label>

              <div className="panel-actions">
                <button className="btn btn--secondary" type="button" onClick={handleStop} disabled={!selectedSessionId}>
                  Stop
                </button>
                <button
                  className="btn btn--secondary"
                  type="button"
                  onClick={handleReplayBest}
                  disabled={!selectedSession?.best_iteration_id}
                >
                  Replay Best
                </button>
              </div>
            </div>

            {selectedSession ? (
              <div className="experiment-card">
                <div>
                  <span className="muted">Status</span>: {selectedSession.status}{" "}
                  {selectedSession.running_task ? <span className="muted">(runner active)</span> : null}
                </div>
                <div>
                  <span className="muted">Best Sharpe</span>: {selectedSession.best_sharpe ?? "—"}{" "}
                  <span className="muted">Best corr</span>: {selectedSession.best_corr ?? "—"}
                </div>
                {selectedSession.last_error ? (
                  <div className="muted error-text">Last error: {selectedSession.last_error}</div>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>

        <div className="experiment-col">
          <div className="experiment-section">
            <h3>Approvals</h3>
            {detailsError ? <p className="muted error-text">{detailsError}</p> : null}
            {pendingProposal ? (
              <div className="experiment-card">
                <div>
                  <strong>Pending proposal #{pendingProposal.id}</strong> (next iteration {pendingProposal.next_iteration})
                </div>
                <pre className="log-viewer">{JSON.stringify(pendingProposal.proposed_updates_json, null, 2)}</pre>
                <div className="panel-actions">
                  <button className="btn" type="button" onClick={() => void handleDecide("approved")}>
                    Approve
                  </button>
                  <button className="btn btn--secondary" type="button" onClick={() => void handleDecide("rejected")}>
                    Reject
                  </button>
                </div>
              </div>
            ) : (
              <p className="muted">No pending proposals.</p>
            )}
          </div>

          <div className="experiment-section">
            <h3>Iterations</h3>
            {iterations.length ? (
              <div className="experiment-table-wrap">
                <table className="experiment-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Status</th>
                      <th>Sharpe</th>
                      <th>Avg corr</th>
                      <th>Run</th>
                    </tr>
                  </thead>
                  <tbody>
                    {iterations.map((it) => (
                      <tr key={it.id}>
                        <td>{it.iteration_index}</td>
                        <td>{it.status}</td>
                        <td>{it.objective_sharpe ?? "—"}</td>
                        <td>{it.objective_corr ?? "—"}</td>
                        <td className="muted">{it.run_dir ? String(it.run_dir).split("/").slice(-1)[0] : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">No iterations recorded yet.</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
