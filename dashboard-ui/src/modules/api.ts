import {
  AlphaTimeseries,
  BacktestRow,
  ConfigDefaultsResponse,
  ConfigListItem,
  ConfigListResponse,
  ConfigPresetValues,
  ConfigPresetsResponse,
  ConfigSavePayload,
  JobActivityResponse,
  JobLogResponse,
  JobStatusResponse,
  LastRunPayload,
  ParamMetaResponse,
  PipelineRunRequest,
  PipelineRunResponse,
  MLModelSpec,
  MLRunDetails,
  MLRunRequest,
  MLRunResponse,
  MLRunSummary,
  CodexModeSettings,
  CodexModeSummary,
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

export async function fetchJobActivity(jobId: string): Promise<JobActivityResponse> {
  return request<JobActivityResponse>(`/api/job-activity/${encodeURIComponent(jobId)}`);
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
  baseline_metrics?: Record<string, unknown> | null;
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
    baselineMetrics: data.baseline_metrics ?? null,
  };
}

interface RunAssetsResponse {
  items: string[];
}

export async function fetchRunAssets(runDir: string, prefix?: string): Promise<string[]> {
  const params = new URLSearchParams({ run_dir: runDir });
  if (prefix) {
    params.set("prefix", prefix);
  }
  const payload = await request<RunAssetsResponse>(`/api/run-assets?${params.toString()}`);
  return Array.isArray(payload.items) ? payload.items : [];
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

interface MlModelsResponse {
  models: Array<{
    id: string;
    label: string;
    description?: string | null;
    presets?: Array<{
      name: string;
      description?: string | null;
      params?: Record<string, unknown>;
    }>;
    default_preset?: string | null;
  }>;
}

export async function fetchMlModels(): Promise<MLModelSpec[]> {
  const data = await request<MlModelsResponse>("/api/ml-lab/models");
  const models = Array.isArray(data.models) ? data.models : [];
  return models.map((model) => ({
    id: model.id,
    label: model.label,
    description: model.description ?? null,
    presets: (model.presets ?? []).map((preset) => ({
      name: preset.name,
      description: preset.description ?? null,
      params: preset.params ?? {},
    })),
    defaultPreset: model.default_preset ?? null,
  }));
}

interface MlRunSummaryResponse {
  name: string;
  path: string;
  status?: string | null;
  best_sharpe?: number | null;
  completed?: number | null;
  total?: number | null;
  started_at?: string | null;
}

export async function fetchMlRuns(limit = 50): Promise<MLRunSummary[]> {
  const data = await request<MlRunSummaryResponse[]>(`/api/ml-lab/runs?limit=${encodeURIComponent(limit)}`);
  return data.map((item) => ({
    name: item.name,
    path: item.path,
    status: item.status ?? null,
    bestSharpe: item.best_sharpe ?? null,
    completed: item.completed ?? null,
    total: item.total ?? null,
    startedAt: item.started_at ?? null,
  }));
}

interface MlRunDetailsResponse {
  name: string;
  path: string;
  summary?: Record<string, unknown> | null;
  results?: Array<Record<string, unknown>> | null;
  spec?: Record<string, unknown> | null;
  meta?: Record<string, unknown> | null;
}

export async function fetchMlRunDetails(runDir: string): Promise<MLRunDetails> {
  const params = new URLSearchParams({ run_dir: runDir });
  const data = await request<MlRunDetailsResponse>(`/api/ml-lab/run-details?${params.toString()}`);
  const results = Array.isArray(data.results) ? data.results : [];
  const asNumber = (value: unknown): number | null => {
    if (value === null || value === undefined) return null;
    if (typeof value === "number") {
      return Number.isFinite(value) ? value : null;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  return {
    name: data.name,
    path: data.path,
    summary: data.summary ?? null,
    results: results.map((item) => ({
      modelId: typeof item.model_id === "string" ? item.model_id : String(item.model_id ?? ""),
      modelLabel: typeof item.model_label === "string" ? item.model_label : null,
      variant: typeof item.variant === "string" ? item.variant : String(item.variant ?? ""),
      preset: typeof item.preset === "string" ? item.preset : null,
      params: (item.params as Record<string, unknown>) ?? null,
      status: typeof item.status === "string" ? item.status : null,
      error: typeof item.error === "string" ? item.error : null,
      Sharpe: asNumber(item.Sharpe),
      AnnReturn: asNumber(item.AnnReturn),
      AnnVol: asNumber(item.AnnVol),
      MaxDD: asNumber(item.MaxDD),
      Turnover: asNumber(item.Turnover),
      IC: asNumber(item.IC),
      NetExposureMean: asNumber(item.NetExposureMean),
      NetExposureMedian: asNumber(item.NetExposureMedian),
      GrossExposureMean: asNumber(item.GrossExposureMean),
    })),
    spec: data.spec ?? null,
    meta: data.meta ?? null,
  };
}

export async function startMlRun(data: MLRunRequest): Promise<MLRunResponse> {
  return request<MLRunResponse>("/api/ml-lab/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
}

export async function stopMlRun(jobId: string): Promise<{ stopped: boolean }> {
  return request<{ stopped: boolean }>(`/api/ml-lab/stop/${encodeURIComponent(jobId)}`, {
    method: "POST",
  });
}

interface CodexSummaryResponse {
  settings?: CodexModeSettings;
  state?: Record<string, unknown>;
  events?: Array<Record<string, unknown>>;
  inbox?: string | null;
  experiments?: string | null;
  session_prompt?: string | null;
  review_needed?: boolean;
  watcher?: {
    running?: boolean;
    pid?: number | null;
    log_file?: string | null;
  } | null;
  updated_at?: string | null;
}

export async function fetchCodexSummary(): Promise<CodexModeSummary> {
  const data = await request<CodexSummaryResponse>("/api/codex-mode/summary");
  const events = Array.isArray(data.events) ? data.events : [];
  const asNumber = (value: unknown): number | null => {
    if (value === null || value === undefined) return null;
    if (typeof value === "number") {
      return Number.isFinite(value) ? value : null;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  return {
    settings: data.settings ?? {
      notify: true,
      review_interval: 3,
      sleep_seconds: 15,
      yolo_mode: false,
      auto_run: false,
      auto_run_command: "codex",
      auto_run_mode: "terminal",
      auto_run_cooldown: 300,
    },
    state: (data.state ?? {}) as CodexModeSummary["state"],
    events: events.map((item) => ({
      ts: typeof item.ts === "string" ? item.ts : null,
      kind: typeof item.kind === "string" ? item.kind : null,
      run_name: typeof item.run_name === "string" ? item.run_name : null,
      run_dir: typeof item.run_dir === "string" ? item.run_dir : null,
      sharpe: asNumber(item.sharpe),
      status: typeof item.status === "string" ? item.status : null,
      label: typeof item.label === "string" ? item.label : null,
    })),
    inbox: data.inbox ?? null,
    experiments: data.experiments ?? null,
    session_prompt: data.session_prompt ?? null,
    review_needed: Boolean(data.review_needed),
    watcher: data.watcher ?? null,
    updated_at: data.updated_at ?? null,
  };
}

export async function updateCodexSettings(payload: Partial<CodexModeSettings>): Promise<CodexModeSummary["settings"]> {
  const data = await request<{ settings?: CodexModeSettings }>("/api/codex-mode/settings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return data.settings ?? {
    notify: true,
    review_interval: 3,
    sleep_seconds: 15,
    yolo_mode: false,
    auto_run: false,
    auto_run_command: "codex",
    auto_run_mode: "terminal",
    auto_run_cooldown: 300,
  };
}

export async function startCodexWatcher(): Promise<{ started: boolean; pid?: number | null; detail?: string | null }> {
  return request<{ started: boolean; pid?: number | null; detail?: string | null }>("/api/codex-mode/start", {
    method: "POST",
  });
}

export async function stopCodexWatcher(): Promise<{ stopped: boolean; detail?: string | null }> {
  return request<{ stopped: boolean; detail?: string | null }>("/api/codex-mode/stop", {
    method: "POST",
  });
}
