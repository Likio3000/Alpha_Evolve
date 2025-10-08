import {
  AlphaTimeseries,
  BacktestRow,
  ConfigDefaultsResponse,
  ConfigListItem,
  ConfigListResponse,
  ConfigPresetValues,
  ConfigPresetsResponse,
  ConfigSavePayload,
  JobLogResponse,
  JobStatusResponse,
  LastRunPayload,
  ParamMetaResponse,
  PipelineRunRequest,
  PipelineRunResponse,
  RunDetails,
  RunLabelPayload,
  RunSummary,
} from "./types";

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    const detail = text || response.statusText;
    throw new Error(`Request failed (${response.status}): ${detail}`);
  }
  if (response.status === 204) {
    return undefined as unknown as T;
  }
  return (await response.json()) as T;
}

export async function fetchRuns(limit = 50): Promise<RunSummary[]> {
  interface RunSummaryResponse {
    name: string;
    path: string;
    label?: string | null;
    sharpe_best?: number | null;
  }

  const data = await request<RunSummaryResponse[]>(`/api/runs?limit=${encodeURIComponent(limit)}`);
  return data.map((item) => ({
    name: item.name,
    path: item.path,
    label: item.label ?? null,
    sharpeBest: item.sharpe_best ?? null,
  }));
}

export async function fetchLastRun(): Promise<LastRunPayload> {
  return request<LastRunPayload>("/api/last-run");
}

export async function fetchBacktestSummary(runDir: string): Promise<BacktestRow[]> {
  const url = `/api/backtest-summary?run_dir=${encodeURIComponent(runDir)}`;
  return request<BacktestRow[]>(url);
}

export async function fetchAlphaTimeseries(runDir: string, alphaId?: string, file?: string): Promise<AlphaTimeseries> {
  const params = new URLSearchParams({ run_dir: runDir });
  if (alphaId) params.set("alpha_id", alphaId);
  if (file) params.set("file", file);
  const url = `/api/alpha-timeseries?${params.toString()}`;
  return request<AlphaTimeseries>(url);
}

export async function startPipelineRun(data: PipelineRunRequest): Promise<PipelineRunResponse> {
  return request<PipelineRunResponse>("/api/pipeline/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/api/job-status/${encodeURIComponent(jobId)}`);
}

export async function fetchJobLog(jobId: string): Promise<JobLogResponse> {
  return request<JobLogResponse>(`/api/job-log/${encodeURIComponent(jobId)}`);
}

export async function stopPipelineJob(jobId: string): Promise<{ stopped: boolean }> {
  return request<{ stopped: boolean }>(`/api/pipeline/stop/${encodeURIComponent(jobId)}`, {
    method: "POST",
  });
}

export async function updateRunLabel(payload: RunLabelPayload): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/api/run-label", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

interface RunDetailsResponse {
  name: string;
  path: string;
  label?: string | null;
  sharpe_best?: number | null;
  summary?: Record<string, unknown>;
  ui_context?: RunDetails["uiContext"];
  meta?: Record<string, unknown> | null;
}

export async function fetchRunDetails(runDir: string): Promise<RunDetails> {
  const params = new URLSearchParams({ run_dir: runDir });
  const data = await request<RunDetailsResponse>(`/api/run-details?${params.toString()}`);
  return {
    name: data.name,
    path: data.path,
    label: data.label ?? null,
    sharpeBest: data.sharpe_best ?? null,
    summary: data.summary ?? null,
    uiContext: data.ui_context ?? null,
    meta: data.meta ?? null,
  };
}

export async function fetchConfigDefaults(): Promise<ConfigDefaultsResponse> {
  return request<ConfigDefaultsResponse>("/api/config/defaults");
}

export async function fetchConfigList(): Promise<ConfigListItem[]> {
  const payload = await request<ConfigListResponse>("/api/config/list");
  return payload.items;
}

export async function fetchConfigPresets(): Promise<Record<string, string>> {
  const payload = await request<ConfigPresetsResponse>("/api/config/presets");
  return payload.presets;
}

interface ConfigPresetParams {
  dataset?: string;
  path?: string;
}

export async function fetchConfigPresetValues(params: ConfigPresetParams): Promise<ConfigPresetValues> {
  const query = new URLSearchParams();
  if (params.dataset) {
    query.set("dataset", params.dataset);
  }
  if (params.path) {
    query.set("path", params.path);
  }
  const qs = query.toString();
  const url = `/api/config/preset-values${qs ? `?${qs}` : ""}`;
  return request<ConfigPresetValues>(url);
}

export async function saveConfigPreset(payload: ConfigSavePayload): Promise<{ saved: string }> {
  return request<{ saved: string }>("/api/config/save", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchPipelineParamsMeta(): Promise<ParamMetaResponse> {
  return request<ParamMetaResponse>("/ui-meta/pipeline-params");
}

export async function fetchEvolutionParamsMeta(): Promise<ParamMetaResponse> {
  return request<ParamMetaResponse>("/ui-meta/evolution-params");
}
