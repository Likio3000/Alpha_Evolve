import React from "react";
import { PipelineJobState } from "../../types";
import { formatDuration, timeAgo } from "./utils";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface ActivityStatusProps {
  status: PipelineJobState["status"];
  connectionState?: "connected" | "retrying" | "stale" | null;
  pctComplete: number | null;
  generationLabel: string | null;
  etaLabel: string | null;
  lastMessage?: string | null;
  lastUpdated?: number | null;
  elapsedSeconds?: number | null;
}

export function ActivityStatus({
  status,
  connectionState,
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

  const statusVariant =
    status === "running" ? "default" :
      status === "error" ? "destructive" :
        status === "complete" ? "secondary" : "outline";

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2 items-center">
        <Badge variant={statusVariant} className="capitalize">{status}</Badge>
        {connectionState && (
          <Badge variant="outline" className={connectionState === "connected" ? "border-green-500 text-green-500" : "text-muted-foreground"}>
            SSE {connectionState}
          </Badge>
        )}
        {generationLabel && <Badge variant="secondary">{generationLabel}</Badge>}
        {etaDisplay && <span className="text-xs text-muted-foreground">{etaDisplay}</span>}
        {elapsedDisplay && <span className="text-xs text-muted-foreground">Elapsed {elapsedDisplay}</span>}
        {updatedLabel && <span className="text-xs text-muted-foreground">Updated {updatedLabel}</span>}
      </div>

      <div className="space-y-1">
        <Progress value={pctComplete || 0} />
        <div className="flex justify-between items-center text-xs">
          <span className="font-medium">{progressLabel}</span>
          {lastMessage && <span className="text-muted-foreground truncate max-w-[300px]">{lastMessage}</span>}
        </div>
      </div>
    </div>
  );
}
