import React from "react";
import { PipelineJobState } from "../../types";
import { formatDuration, timeAgo } from "./utils";

interface ActivityStatusProps {
  status: PipelineJobState["status"];
  pctComplete: number | null;
  generationLabel: string | null;
  etaLabel: string | null;
  lastMessage?: string | null;
  lastUpdated?: number | null;
  elapsedSeconds?: number | null;
}

export function ActivityStatus({
  status,
  pctComplete,
  generationLabel,
  etaLabel,
  lastMessage,
  lastUpdated,
  elapsedSeconds,
}: ActivityStatusProps): React.ReactElement {
  const etaDisplay = etaLabel ? `ETA ${etaLabel}` : null;
  const elapsedDisplay = formatDuration(elapsedSeconds);
  const updatedLabel = timeAgo(lastUpdated);
  const progressLabel =
    pctComplete !== null ? `${pctComplete.toFixed(1)}% complete` : "Waiting for progress updates…";

  return (
    <>
      <div className="pipeline-activity__status-row">
        <div className={`status-badge status-badge--${status}`}>{status}</div>
        {generationLabel ? <span className="pipeline-activity__status-meta">{generationLabel}</span> : null}
        {etaDisplay ? <span className="pipeline-activity__status-meta muted">{etaDisplay}</span> : null}
        {elapsedDisplay ? (
          <span className="pipeline-activity__status-meta muted">Elapsed {elapsedDisplay}</span>
        ) : null}
        {updatedLabel ? <span className="pipeline-activity__status-meta muted">Updated {updatedLabel}</span> : null}
      </div>

      <div className="pipeline-activity__progress">
        <div className="pipeline-activity__progress-bar">
          <div
            className="pipeline-activity__progress-bar-fill"
            style={{ width: `${pctComplete !== null ? pctComplete : 0}%` }}
          />
        </div>
        <div className="pipeline-activity__progress-meta">
          <span>{progressLabel}</span>
          {lastMessage ? <span className="muted">{lastMessage}</span> : null}
        </div>
      </div>
    </>
  );
}
