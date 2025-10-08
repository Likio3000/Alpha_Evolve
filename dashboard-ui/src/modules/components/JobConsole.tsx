import React from "react";
import { PipelineJobState } from "../types";

interface JobConsoleProps {
  job: PipelineJobState | null;
  onStop?: (job: PipelineJobState) => void;
  onCopyLog?: (job: PipelineJobState) => void;
}

export function JobConsole({ job, onStop, onCopyLog }: JobConsoleProps): React.ReactElement {
  if (!job) {
    return (
      <section className="panel">
        <h2>Pipeline Activity</h2>
        <p className="muted">Launch a pipeline run to monitor progress here.</p>
      </section>
    );
  }

  const canStop = job.status === "running" && onStop;

  const handleCopy = () => {
    if (!onCopyLog) return;
    onCopyLog(job);
  };

  return (
    <section className="panel">
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
      <div className={`status-badge status-badge--${job.status}`}>{job.status}</div>
      {job.lastMessage ? <p className="muted">{job.lastMessage}</p> : null}
      <pre className="log-viewer">{job.log || "Waiting for log output…"}</pre>
      {job.sharpeBest !== undefined ? (
        <div className="muted">
          Last reported Sharpe:{" "}
          {job.sharpeBest === null ? "n/a" : job.sharpeBest.toFixed(4)}
        </div>
      ) : null}
    </section>
  );
}
